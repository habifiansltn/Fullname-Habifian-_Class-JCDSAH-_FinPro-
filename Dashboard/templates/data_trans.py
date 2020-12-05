import pandas as pd 

# to read csv file named "samplee" 
df = pd.read_csv(r"E:\Job Connector Data Science Purwadhika\Module-03-Machine Learning\ulikan\Untitled Folder\templates\Adult_Clean.csv") 
df.drop('Unnamed: 0', axis = 1, inplace = True)
df = df.head()

# to save as html file 
# named as "Table" 
df.to_html("dataset.html") 

df_me = pd.read_csv(r"E:\Job Connector Data Science Purwadhika\Module-03-Machine Learning\ulikan\Untitled Folder\templates\df_me_rs.csv")
df_me.rename(columns = {'Unnamed: 0' : 'Model'}, inplace = True)
df_me.set_index('Model', inplace = True)

df_me.to_html('table_me.html')
