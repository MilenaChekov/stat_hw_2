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
    left_border = n * std**2 / chi2.ppf(1 - alpha/2, df=n-1)
      right_border = n * std**2 / chi2.ppf(alpha/2, df=n-1)
    print(left_border, right_border)

<img width="288" alt="Снимок экрана 2022-02-28 в 07 52 22" src="https://user-images.githubusercontent.com/60537367/155926206-d1110ad7-a67c-4c2c-abcb-26ead414510a.png">

    #1.2![Uploading Снимок экрана 2022-02-28 в 00.25.55.png…]()

    left_border = mean - norm.ppf(1 - alpha/2) * std/n**0.5
    right_border = mean + norm.ppf(1 - alpha/2) * std/n**0.5
    print(left_border, right_border)

<img width="278" alt="Снимок экрана 2022-02-28 в 00 26 00" src="https://user-images.githubusercontent.com/60537367/155900585-2eafebca-6335-4848-aad7-aa97a94f6a3c.png">

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
    
<img width="309" alt="Снимок экрана 2022-02-28 в 00 26 05" src="https://user-images.githubusercontent.com/60537367/155900601-d50e0b3f-2314-4098-a73d-a7c6a2ab8818.png">

    #interval = norm.interval(1-alpha, loc=mean, scale=std/CHE.shape[0]**0.5)
    #print(interval)

    #2.1

    nenorm = df.loc[df['Category'] != "0=Blood Donor"]
    nenorm = nenorm['CHE']
    mean = []
    std2 = []
    for i in range(1000):
      random_df = nenorm.sample(n=df.shape[0], replace=True)
      mean.append(np.mean(random_df))
      std2.append(np.std(random_df))

    f = sns.histplot(mean)
    print(f.get_xlim())
    plt.show()
    plt.close()

![Figure_3](https://user-images.githubusercontent.com/60537367/155927438-db136401-1bb3-4818-99ae-e8c84bd09957.png)

<img width="327" alt="Снимок экрана 2022-02-28 в 08 06 58" src="https://user-images.githubusercontent.com/60537367/155927403-06c70d1d-eb5f-46a6-9186-72f16196247f.png">


    f = sns.histplot(std2)
    print(f.get_xlim())
    plt.show()
    plt.close()
    
 ![Figure_4](https://user-images.githubusercontent.com/60537367/155927452-07c0345e-c395-4afc-807a-1f101958c8db.png)
    
<img width="306" alt="Снимок экрана 2022-02-28 в 08 07 02" src="https://user-images.githubusercontent.com/60537367/155927413-a052b5dd-7c8e-4997-95c3-193cc53d23f8.png">

