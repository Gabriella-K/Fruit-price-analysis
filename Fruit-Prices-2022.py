import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_clean_data(filepath):
    """Load and clean the dataset with robust error handling"""
    try:
        df = pd.read_csv(filepath)
       
        if df.empty:
            raise ValueError("Dataset is empty")
            
     
        print("\nMissing values per column:")
        print(df.isnull().sum())
        df.dropna(inplace=True)  
        
        df['Fruit'] = df['Fruit'].str.split(',').str[0].str.strip()
        
    
        df['PricePerCupDollars'] = df['CupEquivalentPrice']
        df['Processed'] = df['Form'].apply(lambda x: 0 if x == 'Fresh' else 1)
        
        df['Form'] = df['Form'].astype('category')
        df['RetailPriceUnit'] = df['RetailPriceUnit'].astype('category')
        
        print("\n✅ Data cleaned successfully!")
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def perform_eda(df):
    """Enhanced EDA with better visualizations"""
    if df is None:
        return
    
    print("\n📊 Basic Statistics:")
    print(df.describe().round(2))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(df['RetailPrice'], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Retail Price Distribution')
    
    sns.boxplot(x='Form', y='PricePerCupDollars', data=df, ax=axes[1])
    axes[1].set_title('Price by Form')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    top10 = df.nlargest(10, 'PricePerCupDollars')
    bottom10 = df.nsmallest(10, 'PricePerCupDollars')
    combined = pd.concat([top10, bottom10])
    
    sns.barplot(x='PricePerCupDollars', y='Fruit', hue='Form', 
               data=combined.sort_values('PricePerCupDollars', ascending=False))
    plt.title('Top/Bottom 10 Fruits by Price per Cup')
    plt.xlabel('Price per Cup ($)')
    plt.ylabel('')
    plt.show()

def preprocess_data(df):
    """Robust preprocessing pipeline"""
    if df is None:
        return None, None
        
    features = ['RetailPrice', 'Yield', 'CupEquivalentSize', 'PricePerCupDollars', 'Processed']
    
    try:
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, features
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return None, None

def find_optimal_clusters(data, max_clusters=8):
    """Enhanced cluster detection with validation"""
    if data is None:
        return 0
        
    wcss = []
    sil_scores = []
    cluster_range = range(2, max_clusters+1)
    
    for n in cluster_range:
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        
        if n > 1:
            sil_scores.append(silhouette_score(data, kmeans.labels_))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(cluster_range, wcss, 'bo-')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters')
    
    ax2.plot(range(2, max_clusters+1), sil_scores, 'go-')
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('Number of Clusters')
    plt.tight_layout()
    plt.show()
    
    optimal_n = np.argmax(sil_scores) + 2
    print(f"🔍 Optimal clusters: {optimal_n} (based on silhouette score)")
    return optimal_n

def apply_kmeans(data, n_clusters, df):
    """Robust clustering with better visualization"""
    if data is None or df is None:
        return None
        
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(data)
      
        pca = PCA(n_components=3)
        components = pca.fit_transform(data)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(components[:,0], components[:,1], components[:,2], 
                           c=df['Cluster'], cmap='viridis', s=50)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('3D Cluster Visualization')
        fig.colorbar(scatter, ax=ax, label='Cluster')
        plt.show()
        
        
        print("\n📌 Cluster Profiles:")
        cluster_stats = df.groupby('Cluster').agg({
            'RetailPrice': ['mean', 'median'],
            'PricePerCupDollars': ['mean', 'median'],
            'Processed': 'mean',
            'Fruit': lambda x: x.value_counts().index[0]
        })
        print(cluster_stats.round(2))
        
     
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster]
            sample_size = min(3, len(cluster_data))
            print(f"\n🍏 Cluster {cluster} (n={len(cluster_data)}):")
            print(cluster_data[['Fruit', 'Form']].sample(sample_size))
            
        return df
        
    except Exception as e:
        print(f"Clustering error: {str(e)}")
        return None

def analyze_price_premiums(df):
    """Enhanced price analysis with statistical testing"""
    if df is None:
        return
        

    fresh = df[df['Processed'] == 0]['PricePerCupDollars']
    processed = df[df['Processed'] == 1]['PricePerCupDollars']
    
    premium = ((processed.mean() - fresh.mean()) / fresh.mean()) * 100
    print(f"\n💲 Price Premium Analysis:")
    print(f"Fresh avg: ${fresh.mean():.2f} | Processed avg: ${processed.mean():.2f}")
    print(f"Processed fruits are {premium:.1f}% more expensive")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Processed', y='PricePerCupDollars', data=df)
    plt.xticks([0, 1], ['Fresh', 'Processed'])
    plt.title('Price Distribution: Fresh vs Processed')
    plt.ylabel('Price per Cup ($)')
    plt.show()

if __name__ == "__main__":
    print("=== 🍎2 Fruit Price Analysis ===")
    
    
    fruit_df = load_and_clean_data('Fruit-Prices-2022.csv')
    
    if fruit_df is not None:
     
        perform_eda(fruit_df)
        
   
        X_scaled, features = preprocess_data(fruit_df)
        if X_scaled is not None:
            optimal_clusters = find_optimal_clusters(X_scaled)
            fruit_df = apply_kmeans(X_scaled, optimal_clusters, fruit_df)
            
           
            if fruit_df is not None:
                analyze_price_premiums(fruit_df)
                
               
                fruit_df.to_csv('Fruit_Clusters.csv', index=False)
                print("\n💾 Results saved to 'Fruit_Clusters.csv'")
    
    print("\n=== Analysis Complete ===")