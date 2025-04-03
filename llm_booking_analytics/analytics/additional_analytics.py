# Code for additional analytics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the hotel bookings dataset"""
    data_path = os.path.join('data', 'hotel_bookings (1).csv')
    return pd.read_csv(data_path)

def analyze_stay_duration(df):
    """Analyze the duration of stays"""
    # Calculate total stay duration
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    
    # Calculate basic statistics
    stay_stats = {
        'mean': round(df['total_stay'].mean(), 2),
        'median': round(df['total_stay'].median(), 2),
        'min': int(df['total_stay'].min()),
        'max': int(df['total_stay'].max()),
        'std': round(df['total_stay'].std(), 2)
    }
    
    # Group by hotel type
    hotel_stay = df.groupby('hotel')['total_stay'].agg(['mean', 'median', 'std']).reset_index()
    hotel_stay = hotel_stay.round(2)
    
    return stay_stats, hotel_stay

def plot_stay_duration(df):
    """Plot the distribution of stay duration"""
    # Calculate total stay duration
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    sns.histplot(data=df, x='total_stay', kde=True)
    plt.title('Distribution of Stay Duration')
    plt.xlabel('Stay Duration (nights)')
    plt.ylabel('Frequency')
    plt.xlim(0, df['total_stay'].quantile(0.95))  # Limit x-axis to 95th percentile for better visualization
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'stay_duration_distribution.png'))
    plt.close()
    
    # Plot by hotel type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='hotel', y='total_stay', data=df)
    plt.title('Stay Duration by Hotel Type')
    plt.xlabel('Hotel Type')
    plt.ylabel('Stay Duration (nights)')
    plt.ylim(0, df['total_stay'].quantile(0.95))  # Limit y-axis to 95th percentile for better visualization
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'stay_duration_by_hotel_type.png'))
    plt.close()

def analyze_adr(df):
    """Analyze the Average Daily Rate (ADR)"""
    # Calculate basic statistics
    adr_stats = {
        'mean': round(df['adr'].mean(), 2),
        'median': round(df['adr'].median(), 2),
        'min': round(df['adr'].min(), 2),
        'max': round(df['adr'].max(), 2),
        'std': round(df['adr'].std(), 2)
    }
    
    # Group by hotel type
    hotel_adr = df.groupby('hotel')['adr'].agg(['mean', 'median', 'std']).reset_index()
    hotel_adr = hotel_adr.round(2)
    
    return adr_stats, hotel_adr

def plot_adr(df):
    """Plot the distribution of ADR"""
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    sns.histplot(data=df, x='adr', kde=True)
    plt.title('Distribution of Average Daily Rate (ADR)')
    plt.xlabel('ADR')
    plt.ylabel('Frequency')
    plt.xlim(0, df['adr'].quantile(0.95))  # Limit x-axis to 95th percentile for better visualization
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'adr_distribution.png'))
    plt.close()
    
    # Plot by hotel type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='hotel', y='adr', data=df)
    plt.title('ADR by Hotel Type')
    plt.xlabel('Hotel Type')
    plt.ylabel('ADR')
    plt.ylim(0, df['adr'].quantile(0.95))  # Limit y-axis to 95th percentile for better visualization
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'adr_by_hotel_type.png'))
    plt.close()

def analyze_special_requests(df):
    """Analyze special requests"""
    # Calculate basic statistics
    special_requests_stats = {
        'mean': round(df['total_of_special_requests'].mean(), 2),
        'median': round(df['total_of_special_requests'].median(), 2),
        'min': int(df['total_of_special_requests'].min()),
        'max': int(df['total_of_special_requests'].max()),
        'std': round(df['total_of_special_requests'].std(), 2)
    }
    
    # Count by number of special requests
    special_requests_counts = df['total_of_special_requests'].value_counts().sort_index().reset_index()
    special_requests_counts.columns = ['num_requests', 'count']
    
    return special_requests_stats, special_requests_counts

def plot_special_requests(df):
    """Plot the distribution of special requests"""
    plt.figure(figsize=(10, 6))
    
    # Create countplot
    sns.countplot(x='total_of_special_requests', data=df)
    plt.title('Distribution of Special Requests')
    plt.xlabel('Number of Special Requests')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'special_requests_distribution.png'))
    plt.close()

def analyze_room_assignment(df):
    """Analyze room assignment vs. reservation"""
    # Check if assigned room type matches reserved room type
    df['room_match'] = df['reserved_room_type'] == df['assigned_room_type']
    
    # Calculate match rate
    match_rate = (df['room_match'].sum() / len(df)) * 100
    
    # Calculate match rate by hotel type
    hotel_match_rate = df.groupby('hotel')['room_match'].agg(['count', 'sum']).reset_index()
    hotel_match_rate['match_rate'] = (hotel_match_rate['sum'] / hotel_match_rate['count']) * 100
    hotel_match_rate.columns = ['hotel', 'total_bookings', 'matched_rooms', 'match_rate']
    hotel_match_rate = hotel_match_rate.round(2)
    
    return round(match_rate, 2), hotel_match_rate

def plot_room_assignment(df):
    """Plot room assignment match rate"""
    # Check if assigned room type matches reserved room type
    df['room_match'] = df['reserved_room_type'] == df['assigned_room_type']
    
    # Calculate match rate by hotel type
    hotel_match_rate = df.groupby('hotel')['room_match'].agg(['count', 'sum']).reset_index()
    hotel_match_rate['match_rate'] = (hotel_match_rate['sum'] / hotel_match_rate['count']) * 100
    hotel_match_rate.columns = ['hotel', 'total_bookings', 'matched_rooms', 'match_rate']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='hotel', y='match_rate', data=hotel_match_rate)
    plt.title('Room Type Match Rate by Hotel Type')
    plt.xlabel('Hotel Type')
    plt.ylabel('Match Rate (%)')
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'room_match_rate.png'))
    plt.close()

def main():
    """Main function to run additional analytics"""
    # Load data
    df = load_data()
    
    # Analyze stay duration
    stay_stats, hotel_stay = analyze_stay_duration(df)
    plot_stay_duration(df)
    
    # Analyze ADR
    adr_stats, hotel_adr = analyze_adr(df)
    plot_adr(df)
    
    # Analyze special requests
    special_requests_stats, special_requests_counts = analyze_special_requests(df)
    plot_special_requests(df)
    
    # Analyze room assignment
    match_rate, hotel_match_rate = analyze_room_assignment(df)
    plot_room_assignment(df)
    
    return {
        'stay_stats': stay_stats,
        'hotel_stay': hotel_stay.to_dict('records'),
        'adr_stats': adr_stats,
        'hotel_adr': hotel_adr.to_dict('records'),
        'special_requests_stats': special_requests_stats,
        'special_requests_counts': special_requests_counts.to_dict('records'),
        'room_match_rate': match_rate,
        'hotel_match_rate': hotel_match_rate.to_dict('records')
    }

if __name__ == "__main__":
    main()