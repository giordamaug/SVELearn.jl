using MLJ
using MLJBase
using Random
using Statistics
using StatsBase
using MLJModelInterface
using CategoricalDistributions

# Define the custom SplitVotingEnsemble model
mutable struct SplitVotingEnsemble <: Probabilistic
    base_model::Any         # Base model instance (e.g., DecisionTreeClassifier(max_depth=5))
    n_voters::Int           # Number of subsets
    voting::String          # Voting strategy: "soft" or "hard"
    random_state::Int       # Random seed
    verbose::Bool           # Verbosity flag
    classifiers::Vector     # Store trained classifiers
    indices::Vector         # Indices of subsets for reproducibility
    label_encoder::Vector{Int}  # Encoded class labels
    class_mapping::Dict     # Mapping between original and encoded labels

    # Custom constructor
    function SplitVotingEnsemble(base_model; n_voters=-1, voting="soft", random_state=42, verbose=false)
        new(base_model, n_voters, voting, random_state, verbose, [], [], Int[], Dict())
    end
end

# Implement the fit method
function MLJModelInterface.fit(model::SplitVotingEnsemble, verbosity::Int, X, y)
    # Label encoding
    labels = unique(y)

    # Count class frequencies and determine number of voters
    all_index_ne, index_e, model.n_voters = set_chuncks_shuffle(y, model.n_voters, random_state=model.random_state, verbose=model.verbose)
    # Split data into majority and minority
    splits = split_and_distribute_remainder(all_index_ne, model.n_voters)

    classifiers = []
    for split in splits
        indx = [split; index_e]
        subX = MLJBase.selectrows(X, indx)
        suby = y[indx]
        # Create an instance of the base model for each subset
        base_model_instance = model.base_model

        mach = machine(base_model_instance, subX, suby, scitype_check_level=0)
        fit!(mach)
        push!(classifiers, mach)
    end

    # Save trained classifiers and indices in the model
    model.classifiers = classifiers
    model.indices = splits

    # MLJ requires returning fitresult, cache, and updated model
    fitresult = nothing
    return (fitresult, nothing, model)
end

# Define a method to handle soft or hard voting
function soft_or_hard_voting(predictions, voting::String)
    if voting == "soft"
        # Soft voting: average the probabilities
        probabilities = []

        classes = levels(predictions[1][1])
        for pred in predictions
            if pred isa CategoricalDistributions.UnivariateFiniteVector
                # Extract the probabilities from each UnivariateFinite in the vector
                ps = [[pdf(d_vector, label) for label in levels(d_vector)] for d_vector in pred]
                push!(probabilities, ps)  # Access .weights for each UnivariateFinite
            else
                push!(probabilities, pred)
            end
        end

        # Stack the probabilities horizontally and compute the mean across classifiers
        prob_matrix = hcat(probabilities...)  # Stack the probabilities horizontally
        mean_probs = mean(prob_matrix, dims=2)
        return [CategoricalDistributions.UnivariateFinite(classes, mean_probs[i, 1], pool=missing) for i in 1:size(mean_probs, 1)]
    elseif voting == "hard"
        # Hard voting: take the mode of predictions (most frequent class)
        return [mode(row) for row in eachrow(reduce(hcat, predictions))]
    else
        throw(ArgumentError("Unknown voting method: $voting"))
    end
end

# Implement the predict method
function MLJModelInterface.predict(model::SplitVotingEnsemble, ::Nothing, Xnew)
    # Collect predictions from each classifier
    predictions = [MLJModelInterface.predict(clf, Xnew) for clf in model.classifiers]

    # Apply the chosen voting method
    return soft_or_hard_voting(predictions, model.voting)
end

# Implement the predict method
function MLJModelInterface.predict_mode(model::SplitVotingEnsemble, ::Nothing, Xnew)
    # Collect predictions from each classifier
    predictions = [MLJModelInterface.predict(clf, Xnew) for clf in model.classifiers]

    # Apply the chosen voting method
    return StatsBase.mode.(soft_or_hard_voting(predictions, model.voting))
end

# Define the keys method for SplitVotingEnsemble
function MLJModelInterface.keys(model::SplitVotingEnsemble)
    return [:base_model, :n_voters, :voting]
end

# Define the parameters method for SplitVotingEnsemble
function parameters(model::SplitVotingEnsemble)
    return Dict(:base_model => model.base_model, :n_voters => model.n_voters, :voting => model.voting)
end

# Supporting functions
function set_chuncks_shuffle(y, n_chunks::Int; random_state::Int=12, verbose::Bool=false)
    # Count class frequencies
    class_counts = countmap(y)
    sorted_classes = sort(collect(class_counts), by=x -> -x[2])
    maxlabel, maxcount = sorted_classes[1]
    secondmaxlabel, secondmaxcount = sorted_classes[2]

    # Split data into majority and minority
    all_index_ne = findall(==(maxlabel), y)
    index_e = findall(!=(maxlabel), y)

    # Determine number of voters
    if n_chunks <= 0
        n_chunks = max(1, round(Int, maxcount / secondmaxcount))
    end

    if verbose
        println("Major label: $maxlabel ($maxcount), 2nd label: $secondmaxlabel ($secondmaxcount)")
        println("Number of voters: $(model.n_voters)")
    end

    Random.seed!(random_state)
    shuffle!(all_index_ne)
    shuffle!(index_e)
    return all_index_ne, index_e, n_chunks
end

function split_and_distribute_remainder(v::Vector, num_chunks::Int)
    if num_chunks <= 0
        throw(ArgumentError("Number of chunks must be a positive integer."))
    end

    n = length(v)
    base_chunk_size = div(n, num_chunks) # Minimum size for each chunk
    remainder = mod(n, num_chunks)      # Remainder to distribute

    # Calculate sizes for each chunk
    chunk_sizes = [base_chunk_size + (i <= remainder ? 1 : 0) for i in 1:num_chunks]

    # Split the vector based on the calculated sizes
    chunks = []
    start = 1
    for size in chunk_sizes
        push!(chunks, v[start:start + size - 1])
        start += size
    end
    return chunks
end  