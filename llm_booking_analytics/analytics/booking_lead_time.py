# Code for lead time analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the hotel bookings dataset"""
    data_path = os.path.join('data', 'hotel_bookings (1).csv')
    return pd.read_csv(data_path)

def calculate_lead_time_statistics(df):
    """Calculate basic statistics for booking lead time"""
    lead_time_stats = {
        'mean': round(df['lead_time'].mean(), 2),
        'median': round(df['lead_time'].median(), 2),
        'min': int(df['lead_time'].min()),
        'max': int(df['lead_time'].max()),
        'std': round(df['lead_time'].std(), 2)
    }
    
    return lead_time_stats

def plot_lead_time_distribution(df):
    """Plot the distribution of booking lead times"""
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    sns.histplot(data=df, x='lead_time', kde=True)
    plt.title('Distribution of Booking Lead Time')
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Frequency')
    plt.xlim(0, df['lead_time'].quantile(0.95))  # Limit x-axis to 95th percentile for better visualization
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lead_time_distribution.png'))
    plt.close()

def analyze_lead_time_by_hotel_type(df):
    """Analyze lead time by hotel type"""
    hotel_lead_time = df.groupby('hotel')['lead_time'].agg(['mean', 'median', 'std']).reset_index()
    hotel_lead_time = hotel_lead_time.round(2)
    
    return hotel_lead_time

def plot_lead_time_by_hotel_type(df):
    """Plot lead time by hotel type"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='hotel', y='lead_time', data=df)
    plt.title('Booking Lead Time by Hotel Type')
    plt.xlabel('Hotel Type')
    plt.ylabel('Lead Time (days)')
    plt.ylim(0, df['lead_time'].quantile(0.95))  # Limit y-axis to 95th percentile for better visualization
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lead_time_by_hotel_type.png'))
    plt.close()

def analyze_lead_time_by_market_segment(df):
    """Analyze lead time by market segment"""
    market_lead_time = df.groupby('market_segment')['lead_time'].agg(['mean', 'median', 'std']).reset_index()
    market_lead_time = market_lead_time.sort_values('mean', ascending=False)
    market_lead_time = market_lead_time.round(2)
    
    return market_lead_time

def plot_lead_time_by_market_segment(df):
    """Plot lead time by market segment"""
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='market_segment', y='lead_time', data=df)
    plt.title('Booking Lead Time by Market Segment')
    plt.xlabel('Market Segment')
    plt.ylabel('Lead Time (days)')
    plt.ylim(0, df['lead_time'].quantile(0.95))  # Limit y-axis to 95th percentile for better visualization
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lead_time_by_market_segment.png'))
    plt.close()

def analyze_lead_time_cancellation_correlation(df):
    """Analyze correlation between lead time and cancellation"""
    # Group by lead time ranges and calculate cancellation rate
    df['lead_time_range'] = pd.cut(df['lead_time'], 
                                  bins=[0, 7, 30, 90, 180, 365, np.inf], 
                                  labels=['0-7 days', '8-30 days', '31-90 days', 
                                          '91-180 days', '181-365 days', '365+ days'])
    
    lead_time_cancellation = df.groupby('lead_time_range')['is_canceled'].agg(['count', 'sum']).reset_index()
    lead_time_cancellation['cancellation_rate'] = (lead_time_cancellation['sum'] / lead_time_cancellation['count']) * 100
    lead_time_cancellation.columns = ['lead_time_range', 'total_bookings', 'canceled_bookings', 'cancellation_rate']
    
    return lead_time_cancellation

def plot_lead_time_cancellation_correlation(lead_time_cancellation):
    """Plot correlation between lead time and cancellation rate"""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='lead_time_range', y='cancellation_rate', data=lead_time_cancellation)
    plt.title('Cancellation Rate by Lead Time Range')
    plt.xlabel('Lead Time Range')
    plt.ylabel('Cancellation Rate (%)')
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lead_time_cancellation_correlation.png'))
    plt.close()

def main():
    """Main function to run the lead time analysis"""
    # Load data
    df = load_data()
    
    # Calculate lead time statistics
    lead_time_stats = calculate_lead_time_statistics(df)
    
    # Plot lead time distribution
    plot_lead_time_distribution(df)
    
    # Analyze lead time by hotel type
    hotel_lead_time = analyze_lead_time_by_hotel_type(df)
    plot_lead_time_by_hotel_type(df)
    
    # Analyze lead time by market segment
    market_lead_time = analyze_lead_time_by_market_segment(df)
    plot_lead_time_by_market_segment(df)
    
    # Analyze correlation between lead time and cancellation
    lead_time_cancellation = analyze_lead_time_cancellation_correlation(df)
    plot_lead_time_cancellation_correlation(lead_time_cancellation)
    
    return {
        'lead_time_stats': lead_time_stats,
        'hotel_lead_time': hotel_lead_time.to_dict('records'),
        'market_lead_time': market_lead_time.to_dict('records'),
        'lead_time_cancellation': lead_time_cancellation.to_dict('records')
    }

if __name__ == "__main__":
    main()