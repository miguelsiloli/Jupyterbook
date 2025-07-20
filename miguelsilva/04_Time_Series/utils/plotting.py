import matplotlib.pyplot as plt

def qq_plot(forecast, df_test):
    qqplot = plt.figure(figsize=(8, 6))
    plt.scatter(forecast, 
                df_test, 
                color='blue', 
                alpha=0.5)
    plt.plot([df_test.min(), 
            df_test.max()], 
            [df_test.min(), 
            df_test.max()], 
            'k--', 
            lw=2)  # Diagonal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    return (qqplot)


def steps_plot(forecast, df_test):
    steps_plot = plt.figure(figsize=(10, 6))
    plt.plot(df_test.index,
            forecast, 
                label='Predicted', 
                marker='o')
    plt.plot(df_test.index, 
            df_test, 
                label='Actual', 
                marker='x')
    plt.xlabel('Number of Steps Predicted')
    plt.ylabel('Value')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.grid(True)
    return (steps_plot)

def error_hist(errors):
    error_hist = plt.figure(figsize=(8, 6))
    plt.hist(errors, 
                bins=30, 
                edgecolor='black')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.grid(True)
    return (error_hist)
