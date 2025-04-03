# Code for geographical distribution analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the hotel bookings dataset"""
    data_path = os.path.join('data', 'hotel_bookings (1).csv')
    return pd.read_csv(data_path)

def analyze_country_distribution(df):
    """Analyze the geographical distribution of hotel bookings"""
    # Count bookings by country
    country_counts = df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    
    # Get top 15 countries
    top_countries = country_counts.head(15)
    
    return top_countries

def plot_country_distribution(top_countries):
    """Plot the geographical distribution of hotel bookings"""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='country', data=top_countries)
    plt.title('Top 15 Countries by Number of Bookings')
    plt.xlabel('Number of Bookings')
    plt.ylabel('Country')
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'country_distribution.png'))
    plt.close()
    
def analyze_hotel_type_by_country(df):
    """Analyze hotel type distribution by country"""
    # Get top 10 countries
    top_countries = df['country'].value_counts().nlargest(10).index.tolist()
    
    # Filter data for top countries
    filtered_df = df[df['country'].isin(top_countries)]
    
    # Create a crosstab of country vs hotel type
    hotel_by_country = pd.crosstab(filtered_df['country'], filtered_df['hotel'])
    
    return hotel_by_country

def plot_hotel_type_by_country(hotel_by_country):
    """Plot hotel type distribution by country"""
    plt.figure(figsize=(12, 8))
    hotel_by_country.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Hotel Type Distribution by Country (Top 10 Countries)')
    plt.xlabel('Country')
    plt.ylabel('Number of Bookings')
    plt.legend(title='Hotel Type')
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hotel_type_by_country.png'))
    plt.close()

def main():
    """Main function to run the geographical distribution analysis"""
    # Load data
    df = load_data()
    
    # Analyze country distribution
    top_countries = analyze_country_distribution(df)
    plot_country_distribution(top_countries)
    
    # Analyze hotel type by country
    hotel_by_country = analyze_hotel_type_by_country(df)
    plot_hotel_type_by_country(hotel_by_country)
    
    return {
        'top_countries': top_countries.to_dict('records'),
        'hotel_by_country': hotel_by_country.to_dict()
    }

if __name__ == "__main__":
    main()