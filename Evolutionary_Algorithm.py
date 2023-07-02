import numpy as np

# number of iterations in the Step_2
step_2_iterations = 4
# for every matrix containing samples in the way (t_Start, t_End) the columns are then numerated:
t_start_column = 0
t_end_column = 1
# Possible offset of the range (T_Start, t_0) for the creation of samples in step_1
offset = 10
# Replace for zero to be able to calculate relative_fitness_value (reciprocal of value)
mindestwert	= 0.0001

# T_Start: Instant when pre-crash recording time starts.
T_Start	= -5000
# t_0: Instant when pre-crash recording time ends.  
# As t_0 is the time when the recording algorithm start, it is normally set to zero.
t_0 = 0
# T: Pre-crash recording time established in the regulation.
T = (T_Start, 0)
# popsize: Population size in the Evolutive Algorithm.  Number of test samples created with the algorithm
popsize	= 100
# lambda_new: New generated samples in each iteration
lambda_new = 2
# The value obtained from the minimum is divided by 1000 so that the value of V_1 is comparable with the of V_2 (next criteria).
V_1_scale = 0.001

# For the fitness function:
a	= 1  # Factor to weight | t_Start − E_i [ t_Start ] | in V_1
b	= 3  # Factor to weight | t_End − E_i [ t_End ] | in V_1
c	= 5  # Factor to weight V_2 over V_1

# For the mutation function:
mu = 0  # Mean in normal distribution: for the standard normal distribution set in zero
sigma = 5 # Standard deviation in normal distribution: It is also known as the scale factor

# error_hit_set: (Fehlertreff in Excel sheet) is the set with the samples, that have discovered an
# error within the test procedure.  At the beginning of the whole procedure the set is empty, but
# as the procedure discover errors in testing, it will be filled with them.
# error_hit_set = np.array([[-4025, -300],[-3213, -1232]])
error_hit_set = np.array([])

def initialize_samples_array():
    # Generate random values for the t_start values in the range [0,1]
    t_start = np.random.rand(popsize, 1)
    
    # Generate random values for the t_End values in the range [0,t_start]
    t_End = np.random.rand(popsize, 1) * t_start
    
    # Concatenate the columns to create the array of samples in the way (t_start, t_End)
    array = np.concatenate((t_start, t_End), axis=1)
    
    # Establish the array in the range from T_start to t_0 plus the offset and round to have only integers 
    array = offset + np.rint(array * (T_Start - 2*offset))
    
    return array
  
def fitness_function(samples, error_hit_set):
    res_V_1 = V_1(samples, error_hit_set) + c*V_2(samples)

    return res_V_1

def V_1(samples_array, error_hit_set):
    if error_hit_set.size == 0:  # if error_hit_set is empty
        return np.zeros(samples_array.shape[0]) # return an array filled with zeros

    # weight (t_Start, E_i[t_Starμ]) and t_End, E_i[t_End] in the samples and error_hit_set matrices 
    # with a and b correspondingly before the diffrence comparison between them 
    weights = np.array([[a,b]])
    samples_weigthed = samples_array * weights
    error_hit_set_weigthed = error_hit_set * weights
    
    # Calculate the absolute difference between each row in samples_weigthed and all rows in error_hit_set
    diff = np.abs(samples_weigthed[:, None, :] - error_hit_set_weigthed)

    # Sum the members for each sample case
    sum_evaluation = diff.sum(axis=2)
    
    # return the minimum for each case multiplied by V_1_scale
    return np.min(sum_evaluation, axis=1) * V_1_scale

def V_2(samples_array):
    # Get the t_end of samples
    t_end = samples_array[:, t_end_column]

    # If t_end>t_0 we return the difference, otherwise we return T_Start - t_end,
    # this last leaves negative results when T_Start <= t_end <= t_0, namely, when t_end is contained in T, thus:
    #
    # distance = np.where(t_end>t_0, t_end - t_0, T_Start - t_end)
    #
    # But as t_0 is set at zero then t_end - t_0 can be just t_end
    distance = np.where(t_end>t_0, t_end, T_Start - t_end)
    
    # We return only the positive numbers, as they are the ones not contained in the range [T_Start,t_0], 
    # othewise we return zero
    return distance.clip(min=0)

def relative_fitness_value(samples_score):
    # replace zeros in samples_score with "mindestwert" value that can be passed as an argument to np.reciprocal
    samples_score = np.where(samples_score == 0, mindestwert, samples_score)
    
    reciprocal = np.reciprocal(samples_score)
    reciprocal_sum = np.sum(reciprocal)
    result = reciprocal * (1 / reciprocal_sum)
    
    return result

def select_offspring(bins, samples, lambda_new):
    x = np.random.rand(lambda_new)
    #x = np.array([0.764859213377431, 0.615431627867042])  # random numbers used in the example
    #bins = np.array([0.25062546, 0.3624107,  0.25580673, 0.13115711])  # bins (p) in the example
    bins_cumulative = np.cumsum(bins)
    selection = np.digitize(x, bins_cumulative)
    samples_selected = samples[selection,:]
    
    return samples_selected

def select_new_generation(matrix, scores):
    # Sort the rows based on the negation of scores.  
    # As negate an array, the lowest elements become the highest elements
    sorted_indices = np.argsort(-scores)
    sorted_matrix = matrix[sorted_indices]
    
    # Select the top popsize samples
    best_samples = sorted_matrix[-popsize:]
    
    return best_samples

# 1. Random test samples creation: 
#  In order to maximize the exploration of possible errors, new random samples will be created, 
#  so that every time the test is executed, it will take new values to test. The restriction 
#  for the creation of test samples is the time range where the end of the trivial event can
#  take place (T_Start, t_0) => (T_Start, 0).

def step_1():
    # samples = initialize_samples_array()
    samples = np.array([[-2420,-259],[-4338,-594],[-3616,-728],[-1311,-104]])

    return samples

# 2. Test execution and following test sample creation process.

def step_2(samples):
    # 2.1. Evaluation:
    #  After the first test execution, the test results are evaluated within the fitness function 
    #  [Equation 6].

    samples_score = fitness_function(samples, error_hit_set)

    # 2.2. Offspring:
    #  Then, by means of Roulette wheel selection (RWS) method the samples to generate new samples
    #  (offspring) are selected.

    p = relative_fitness_value(samples_score)  # The relative_fitness_value of each sample based on its sample_score
    offspring_selection = select_offspring(p, samples, lambda_new)  # A lambda_new sized samples array based on RWS method
    
    # 2.3. Mutation:
    #  In this case the "Recombination" phase does not take place, so directly the new samples are
    #  mutated by means of Normal mutation function [Equation 8] for each
    #  property of the new samples (t_Start, t_End ).

    mutation = np.round(np.random.normal(mu, sigma, size=(lambda_new, 2))).astype(int)
    offspring = offspring_selection + mutation

    # 2.4. Mutated Samples Evaluation:
    #  The new samples are evaluated within the fitness function.
    offspring_score = fitness_function(offspring, error_hit_set)

    # 2.5. New generation:
    #  The best (popsize) evaluated test samples from the set of current and new test samples are
    #  taken for the next iteration of the process.

    new_samples = np.concatenate((samples, offspring))
    new_scores = np.concatenate((samples_score, offspring_score))
    new_generation = select_new_generation(new_samples, new_scores)

    return new_generation

# Then, the EA algorithm is:

test_samples = step_1()
print(test_samples)
# iterations of step_2
for _ in range(step_2_iterations):
    test_samples = step_2(test_samples)
    
print(test_samples)