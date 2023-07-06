#Import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import time

#Uniform distribution
import random

def generate_uniform_data(mean, std_dev, shape):
    min_val = mean - std_dev * 3
    max_val = mean + std_dev * 3

    data = np.random.uniform(min_val, max_val, shape)
    return data


# Generare date panel
def generare_date_panel(N, T, sigma_err, sigma_alpha, beta1, beta2 ):
    
    # Generăm variabilele X1 și X2
    X1 = generate_uniform_data(3, 2, N*T)
    X1 = np.array(X1).reshape(N*T,)
    X2 = np.random.normal(36, 25, N*T)
    X2 = np.array(X2).reshape(N*T,)

    beta0 = np.random.RandomState(seed=42).rand(N,)*10

    # Generăm efectele aleatoare individuale
    alpha = np.kron(np.eye(N),np.ones((T, 1)))

    # Generăm erorile
    epsilon = generate_uniform_data(0, sigma_err, N*T)
    epsilon = np.array(epsilon).reshape(N*T, 1)

    arg1 = np.array(beta0.repeat(T)).reshape(N*T,1)
    arg2 = np.array(beta1*X1).reshape(N*T, 1)
    arg3 = np.array(beta2*X2).reshape(N*T, 1)
    arg4 = np.array(np.random.normal(0,sigma_alpha, N).reshape(N,1))
  

    # Generăm Y cu efecte aleatoare
    Y = arg1+ arg2+ arg3 + np.dot(alpha,arg4) + epsilon
    Y = np.array(Y).reshape(N*T,)
    
    # Cream panelul folosind pandas DataFrame
    data = pd.DataFrame({'id': np.repeat(np.arange(N), T),
                         'time': np.tile(np.arange(T), N),
                         'Y': Y,
                         'X1': X1,
                         'X2': X2                 
                        })

    return data


#PooledOLS
def PooledOLS():
    global beta1_array, beta2_array
    Y = df_panel.iloc[:,2]
    X = df_panel.iloc[:,3:]

    pooled_olsr_model = sm.OLS(Y, sm.add_constant(X))    
    pooled_olsr_model_results = pooled_olsr_model.fit()
    
    # Extragerea coeficientilor beta1 si beta2
    beta1 = pooled_olsr_model_results.params[1]  # Beta 1 coefficient
    beta2 = pooled_olsr_model_results.params[2]  # Beta 2 coefficient
    
    #Adaugam coeficientii la array-uri
    beta1_array = np.append(beta1_array, beta1)
    beta2_array = np.append(beta2_array, beta2)

#Within
def Within():
    global beta1_arrayb, beta2_arrayb

    #Define x si y
    Y = df_panel.iloc[:,2]
    X = df_panel.iloc[:,3:]

    y_var_name = 'Y'
    X_var_names = ['X1','X2']

    unit_col_name=df_panel.columns[0]

    #Create the dummy variables, one for each country
    df_dummies = pd.get_dummies(df_panel[unit_col_name], prefix = 'id')

    # Join the dummies Dataframe to the panel data set:
    df_panel_with_dummies = df_panel.join(df_dummies)
    
    # Construct the regression equation
    lsdv_expr = f"{y_var_name} ~ {' + '.join(X_var_names)} + {' + '.join(df_dummies.columns[:-1])}"
    
    # Build and train an LSDV model on the panel data containing dummies:
    lsdv_model = smf.ols(formula=lsdv_expr, data=df_panel_with_dummies)
    lsdv_model_results = lsdv_model.fit()

    # Extragerea coeficientilor beta1 si beta2
    beta1 = lsdv_model_results.params[1]  # Beta 1 coefficient
    beta2 = lsdv_model_results.params[2]  # Beta 2 coefficient

    
    #Adaugam coeficientii la array-uri
    beta1_arrayb = np.append(beta1_arrayb, beta1)
    beta2_arrayb = np.append(beta2_arrayb, beta2)

# Formatare date obtinute in tabele
def tableData(data):
    global N_values,T_values
    
    # Impartim datele in cate N-uri avem
    arrays = np.split(data, len(N_values))

    # Concatenam datele vertical
    combined_array = np.vstack(arrays)

    # Creem nemele coloanelor in functie de T
    column_names = ['T=' + str(t) for t in T_values]

    # Transformam vectorul combinat intr-un pandas dataFrame
    df = pd.DataFrame(combined_array, columns=column_names)

    # Adaugam o noua prima coloana cu variatia N-ului
    df.insert(0, 'N\T', ['N=' + str(n) for n in N_values])

    return df

# Generare grafice cu datele din tabele
def plot_line_plots(data):
    # Extragere date din tabel
    plot_data = data.iloc[:, 1:].values
    row_names = data.iloc[:, 0].values

    # Setam figura si axele
    fig, ax = plt.subplots()

    # Iteram fiecare rand din date
    for i in range(len(plot_data)):
        x = np.arange(len(plot_data[i]))  # x-axis values
        y = plot_data[i]  # y-axis values

        # Plotul
        ax.plot(x, y, label=row_names[i])

    # Setam etichetele pentru axa x
    x_labels = data.columns[1:]
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)

    # Mutam legenda inafara plotului
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Setam etichetele acelor si titlul
    ax.set_xlabel('Perioade de timp')
    ax.set_ylabel('')
    ax.set_title('')

    # Ajustam layoutul
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    # Afisare
    plt.show()

# Generare tabele combinate
def plot_line_comb(data1, data2):
    # Extract data from tables
    plot_data1 = data1.iloc[:, 1:].values
    plot_data2 = data2.iloc[:, 1:].values
    row_names = data1.iloc[:, 0].values
    column_names = data1.columns[1:]

    num_rows = plot_data1.shape[0]

    # Determine the number of subplots
    num_cols = 2
    num_subplots = num_rows // num_cols + num_rows % num_cols

    # Create subplots
    fig, axs = plt.subplots(num_subplots, num_cols, figsize=(10, 5))

    # Flatten the axs array if num_subplots is 1
    if num_subplots == 1:
        axs = [axs]

    # Iterate through each row and create line plots
    for i, row in enumerate(range(num_rows)):
        row_data1 = plot_data1[row, :]
        row_data2 = plot_data2[row, :]
        ax = axs[i // num_cols][i % num_cols]
        ax.plot(column_names, row_data1, label='PooledOLS')
        ax.plot(column_names, row_data2, label='Within')
        ax.set_title(row_names[row])
        ax.legend()

    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()


## Main ##

# Definim parametri
N_values = [10, 50, 90, 130, 170, 210]  # number of individuals
T_values = [10, 50, 90, 130, 170, 210]  # time periods for each panel
sigma_err = 0.5
sigma_alpha = 3
beta1 = 7
beta2 = 4


#Vectori cu toate valorile urmate simularilor monte carlo : PooledOLS
mean_beta1_general = np.array([])
mean_beta2_general = np.array([])
std_deviation_beta1_general = np.array([])
std_deviation_beta2_general = np.array([])

#Vectori cu toate valorile urmate simularilor monte carlo : Within
mean_beta1_generalb = np.array([])
mean_beta2_generalb = np.array([])
std_deviation_beta1_generalb = np.array([])
std_deviation_beta2_generalb = np.array([])

# Start the timer
start_time = time.time()


#Loop over individual values
for N in N_values:
    # Loop over time period values
    for T in T_values:
        
        # Vectori cu valorile lui Beta1, Beta2 PooledOLS
        beta1_array = np.array([])
        beta2_array = np.array([])
        
        # Vectori cu valorile lui Beta1, Beta2 Within
        beta1_arrayb = np.array([])
        beta2_arrayb = np.array([])

        # Create panel for each time period value
        for i in range(200):
        
            # Generate panel data for each individual
            df_panel = generare_date_panel(N, T, sigma_err, sigma_alpha , beta1, beta2)

            #PooledOLS (extrag beta-urile intr-un vector global)
            PooledOLS()
            #Within (extrag beta-urile intr-un vector global)
            Within()

          
        #Calculez media si standard deviation pentru Beta PooledOLS
        mean_value_beta1 = np.mean(beta1_array)
        mean_value_beta2 = np.mean(beta2_array)
        std_deviation_beta1 = np.std(beta1_array)
        std_deviation_beta2 = np.std(beta2_array)

        #Salvez valorile in liste/ arrays globale pentru PooledOLS
        mean_beta1_general = np.append(mean_beta1_general,mean_value_beta1)
        mean_beta2_general = np.append(mean_beta2_general,mean_value_beta2)
        std_deviation_beta1_general = np.append(std_deviation_beta1_general,std_deviation_beta1)
        std_deviation_beta2_general = np.append(std_deviation_beta2_general,std_deviation_beta2)

        #Calculez media si standard deviation pentru Beta Within
        mean_value_beta1b = np.mean(beta1_arrayb)
        mean_value_beta2b = np.mean(beta2_arrayb)
        std_deviation_beta1b = np.std(beta1_arrayb)
        std_deviation_beta2b = np.std(beta2_arrayb)
        #Salvez valorile in liste/ arrays globale pentru Within
        mean_beta1_generalb = np.append(mean_beta1_generalb,mean_value_beta1b)
        mean_beta2_generalb = np.append(mean_beta2_generalb,mean_value_beta2b)
        std_deviation_beta1_generalb = np.append(std_deviation_beta1_generalb,std_deviation_beta1b)
        std_deviation_beta2_generalb = np.append(std_deviation_beta2_generalb,std_deviation_beta2b)

    print("Done N = ", N)
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Print the running time
    print(f"Elapsed time: {elapsed_time} seconds")        

#Results PooledOLS
mean_beta1 = tableData(mean_beta1_general)
print("Beta1 mediu: \n")
print( mean_beta1)
mean_beta2 = tableData(mean_beta2_general)
print("Beta2 mediu: \n", mean_beta2)
std_beta1 = tableData(std_deviation_beta1_general)
print("Deviatia standard Beta1: \n", std_beta1)
std_beta2 = tableData(std_deviation_beta2_general)
print("Deviatia standard Beta2: \n", std_beta2)

#Results Within
mean_beta1b = tableData(mean_beta1_generalb)
print("Beta1 mediu: \n")
print( mean_beta1b)
mean_beta2b = tableData(mean_beta2_generalb)
print("Beta2 mediu: \n", mean_beta2b)
std_beta1b = tableData(std_deviation_beta1_generalb)
print("Deviatia standard Beta1: \n", std_beta1b)
std_beta2b = tableData(std_deviation_beta2_generalb)
print("Deviatia standard Beta2: \n", std_beta2b)

#Ploturi PooledOLS
plot_line_plots(mean_beta1)
plot_line_plots(mean_beta2)
plot_line_plots(std_beta1)
plot_line_plots(std_beta2)

#Ploturi Within
plot_line_plots(mean_beta1b)
plot_line_plots(mean_beta2b)
plot_line_plots(std_beta1b)
plot_line_plots(std_beta2b)

#Ploturi combinate
plot_line_comb(mean_beta1,mean_beta1b)
plot_line_comb(mean_beta2,mean_beta2b)
plot_line_comb(std_beta1,std_beta1b)
plot_line_comb(std_beta2,std_beta2b)