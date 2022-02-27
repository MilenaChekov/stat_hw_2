# stat_hw_2

  import pandas as pd
  import numpy as np
  import seaborn as sns
  import matplotlib.pyplot as plt
  from scipy.stats import norm, chi2

  df = pd.read_csv('/Users/milena/PycharmProjects/st/HepatitisCdata.csv')

  healthy = df.loc[df['Category'] == '0=Blood Donor']
  CHE = healthy['CHE']
  sns.histplot(CHE, stat='density')
  plt.show()
  #x0, x1 = f.get_xlim()
  #x_pdf = np.linspace(x0, x1)
  #y_pdf = norm.pdf(x_pdf)
  #f.plot(x_pdf, y_pdf)
  #fig = f.get_figure()
  #fig.savefig('donor_hist.png')
  plt.close()
  
![Figure_1](https://user-images.githubusercontent.com/60537367/155900396-045deed1-a32d-41ff-bde4-eea945d4e49d.png)

  mean = np.mean(CHE)
  std = np.std(CHE)
  alpha = 0.05
  n = CHE.shape[0]

  #1.1
  left_border = (n - 1) * std / chi2.ppf(1 - alpha/2, df=n-1)
  right_border = (n - 1) * std / chi2.ppf(alpha/2, df=n-1)
  print(left_border, right_border)

  #1.2
  left_border = mean - norm.ppf(1 - alpha/2) * std/n**0.5
  right_border = mean + norm.ppf(1 - alpha/2) * std/n**0.5
  print(left_border, right_border)

  #interval = norm.interval(1-alpha, loc=mean, scale=std/CHE.shape[0]**0.5)
  #print(interval)

  #print(np.quantile(std, 0.025), np.quantile(std, 0.975))

  CHE = df['CHE']
  mean = np.mean(CHE)
  std = np.std(CHE)
  n = CHE.shape[0]

  sns.histplot(CHE, stat='density')
  plt.show()
  plt.close()
  
![Figure_2](https://user-images.githubusercontent.com/60537367/155900409-0ea76c75-b6cf-4118-8eda-302e596d2014.png)


  left_border = mean - norm.ppf(1 - alpha/2) * std/n**0.5
  right_border = mean + norm.ppf(1 - alpha/2) * std/n**0.5
  print(left_border, right_border)

  #interval = norm.interval(1-alpha, loc=mean, scale=std/CHE.shape[0]**0.5)
  #print(interval)

  #2.1

  nenorm = df.loc[df['Category'] != "0=Blood Donor"]
  nenorm = nenorm['CHE']
  mean = []
  std = []
  for i in range(1000):
      random_df = nenorm.sample(n=df.shape[0], replace=True)
      mean.append(np.mean(random_df))
      std.append(np.std(random_df))

  sns.histplot(mean)
  plt.show()
  plt.close()
  
![Figure_3](https://user-images.githubusercontent.com/60537367/155900438-efe27afd-930a-4df5-816c-6e9aac3398d4.png)

  sns.histplot(std)
  plt.show()
  plt.close()

![Figure_4](https://user-images.githubusercontent.com/60537367/155900443-eb2f117e-6f58-4b52-a8a9-f19c7354091f.png)
