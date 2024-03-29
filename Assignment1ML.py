import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Yazeed Hamdan 1201133
# part 1
file_path = r'C:\Users\hp\Desktop\ML\cars.csv'  #Load a CSV file containing cars data
Cars_DataFrame = pd.read_csv(file_path)
# Checking the number of features and examples
total_features  = Cars_DataFrame.shape[1]
total_examples  = Cars_DataFrame.shape[0]
print("\n////////////////// Part 1 /////////////////////////////\n")
# Prints the number of features and examples in the dataset
print("Number of Features:", total_features)
print("Number of Examples:", total_examples )

# part 2
print("\n////////////////// Part 2 /////////////////////////////\n")
missing_values = Cars_DataFrame.isnull().sum() # Checking for missing values in each feature
print(missing_values) # print the number of missing values in each feature

# part 3
print("\n////////////////// Part 3 /////////////////////////////\n")
horsepower_median = Cars_DataFrame['horsepower'].median() # For horsepower the median was used
origin_mode = Cars_DataFrame['origin'].mode()[0] # For origin the mode was used
#fill missing values 
for index, row in Cars_DataFrame.iterrows():
    if pd.isnull(row['horsepower']):
        Cars_DataFrame.at[index, 'horsepower'] = horsepower_median
    if pd.isnull(row['origin']):
        Cars_DataFrame.at[index, 'origin'] = origin_mode

# Calculate the number of missing values
updated_missing_values = Cars_DataFrame.isnull().sum()
print(updated_missing_values)
print("\n")
print("Median:",horsepower_median)
print("Mode:",origin_mode)

# Part 4 
plt.figure(figsize=(8, 5)) # size of the figure
sns.boxplot(x='origin', y='mpg', data=Cars_DataFrame, color='orange')
# Set the title and x,y labels in the figure
plt.title('MPG by Country of Origin') 
plt.xlabel('Country')
plt.ylabel('Miles Per Gallon (MPG)')
plt.show()

# part 5
plt.figure(figsize=(10, 5)) # figure size 
# Histogram for acceleration
plt.subplot(1, 3, 1)
# creates a histogram with a Kernel Density Estimate (kde)
# To provide a smooth curve that gives a visualization of the shape of the data distribution
sns.histplot(Cars_DataFrame['acceleration'], kde=True, color='red')
plt.title('Histogram of Acceleration')
# Histogram for horsepower
plt.subplot(1, 3, 2)
sns.histplot(Cars_DataFrame['horsepower'], kde=True, color='blue')
plt.title('Histogram of Horsepower')
# Histogram for mpg
plt.subplot(1, 3, 3)
sns.histplot(Cars_DataFrame['mpg'], kde=True, color='green')
plt.title('Histogram of MPG')
plt.tight_layout() # to adjusts the layout
plt.show()

# part 6
print("\n////////////////// Part 6 /////////////////////////////\n")
# Calculate skewness for each feature using skew()
skewness_acceleration = Cars_DataFrame['acceleration'].skew()
skewness_horsepower = Cars_DataFrame['horsepower'].skew()
skewness_mpg = Cars_DataFrame['mpg'].skew()
# Compare absolute skewness values to find the closest to zero
if abs(skewness_acceleration) < abs(skewness_horsepower) and abs(skewness_acceleration) < abs(skewness_mpg):
    closest_to_gaussian = 'acceleration'
    skew_value = skewness_acceleration
elif abs(skewness_horsepower) < abs(skewness_acceleration) and abs(skewness_horsepower) < abs(skewness_mpg):
    closest_to_gaussian = 'horsepower'
    skew_value = skewness_horsepower
else:
    closest_to_gaussian = 'mpg'
    skew_value = skewness_mpg

# Print the feature with skewness value closest to zero
print(f"\nThe feature with a distribution most similar to a Gaussian is: {closest_to_gaussian} with a skewness of {skew_value}")
print("skewness of horsepower =",skewness_horsepower)
print("skewness of mpg =",skewness_mpg)




# part 7
plt.figure(figsize=(10, 6)) # figure size
sns.scatterplot(x='horsepower', y='mpg', data=Cars_DataFrame)
# Set the title and x,y labels for the figure
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('MPG VS Horsepower')
plt.show()

# part 8
# Adding x0 = 1 to the dataset for the intercept
X = Cars_DataFrame['horsepower']
X = np.c_[np.ones(X.shape[0]), X]  # Adding the intercept term
y = Cars_DataFrame['mpg'] # Target variable
VectorOfW = np.linalg.inv(X.T @ X) @ X.T @ y # Closed form solution
# Plotting the scatter plot and the learned regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='horsepower', y='mpg', data=Cars_DataFrame)
# Plotting the regression line
x_values = np.array([Cars_DataFrame['horsepower'].min(), Cars_DataFrame['horsepower'].max()])
y_values = VectorOfW[0] + VectorOfW[1] * x_values
plt.plot(x_values, y_values, color="red")
plt.title('MPG VS Horsepower with Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()
print("\n////////////////// Part 8 /////////////////////////////\n")
print ("parameters values:", VectorOfW)

# Part 9
X = Cars_DataFrame['horsepower'].values
X_quad = np.column_stack((np.ones(X.shape), X, X**2)) # prepare the feature matrix for a quadratic regression model
VectorOfW_quad = np.linalg.inv(X_quad.T @ X_quad) @ X_quad.T @ y # Simplified Quadratic Regression Model
plt.figure(figsize=(10, 6))
sns.scatterplot(x='horsepower', y='mpg', data=Cars_DataFrame)
# Quadratic Regression Line calculations 
x_values_quad = np.linspace(X.min(), X.max(), 400)
# calculate the y coordinates for the quadratic regression
y_values_quad = VectorOfW_quad[0] + VectorOfW_quad[1] * x_values_quad + VectorOfW_quad[2] * x_values_quad**2
plt.plot(x_values_quad, y_values_quad, color = 'red') # plot quadratic regression line
plt.title('MPG VS Horsepower with Quadratic Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()
print("\n////////////////// Part 9 /////////////////////////////\n")
print("parameters values:",VectorOfW_quad)

# part 10
# Normalize the horsepower data using Z-score normalization
HP_mean = Cars_DataFrame['horsepower'].mean() #calculate the mean
HP_std = Cars_DataFrame['horsepower'].std() #calculate standard deviation
X = (Cars_DataFrame['horsepower'].values - HP_mean) / HP_std
X = np.c_[np.ones(X.shape[0]), X]  # Adding intercept term
y = Cars_DataFrame['mpg'].values
# Initialize Parameters for Gradient Descent
Vector_W = np.array([0.0, 0.0])  # Both w0 and w1 set to 0
learningRate = 0.1 
iterations = 10000  # Number of iterations

# Gradient Descent Algorithm
for _ in range(iterations):
    predictions = X @ Vector_W
    Errors = predictions - y
    gradients = X.T @ Errors / len(X)
    Vector_W -= learningRate * gradients
plt.figure(figsize=(10, 6)) # Plotting the Results
sns.scatterplot(x='horsepower', y='mpg', data=Cars_DataFrame)
# Regression Line
x_values = np.linspace(Cars_DataFrame['horsepower'].min(), Cars_DataFrame['horsepower'].max(), 100)
x_standardized = (x_values - HP_mean) / HP_std
x_standardized = np.c_[np.ones(x_standardized.shape[0]), x_standardized]
y_values = x_standardized @ Vector_W
plt.plot(x_values, y_values, color='red')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('MPG VS Horsepower with Regression Line (Gradient Descent Algorithm)')
plt.show()
print("\n////////////////// Part 10 /////////////////////////////\n")
print("parameters values :", Vector_W)
print("\n")