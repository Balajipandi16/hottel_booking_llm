# Code for cancellation rate analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the hotel bookings dataset"""
    data_path = os.path.join('data', 'hotel_bookings (1).csv')
    return pd.read_csv(data_path)

def calculate_overall_cancellation_rate(df):
    """Calculate the overall cancellation rate"""
    total_bookings = len(df)
    canceled_bookings = df['is_canceled'].sum()
    cancellation_rate = (canceled_bookings / total_bookings) * 100
    
    return {
        'total_bookings': total_bookings,
        'canceled_bookings': int(canceled_bookings),
        'cancellation_rate': round(cancellation_rate, 2)
    }

def analyze_cancellation_by_hotel_type(df):
    """Analyze cancellation rates by hotel type"""
    hotel_cancellation = df.groupby('hotel')['is_canceled'].agg(['count', 'sum'])
    hotel_cancellation['cancellation_rate'] = (hotel_cancellation['sum'] / hotel_cancellation['count']) * 100
    hotel_cancellation = hotel_cancellation.reset_index()
    hotel_cancellation.columns = ['hotel', 'total_bookings', 'canceled_bookings', 'cancellation_rate']
    
    return hotel_cancellation

def plot_cancellation_by_hotel_type(hotel_cancellation):
    """Plot cancellation rates by hotel type"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='hotel', y='cancellation_rate', data=hotel_cancellation)
    plt.title('Cancellation Rate by Hotel Type')
    plt.xlabel('Hotel Type')
    plt.ylabel('Cancellation Rate (%)')
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cancellation_by_hotel_type.png'))
    plt.close()

def analyze_cancellation_by_month(df):
    """Analyze cancellation rates by month"""
    # Order months chronologically
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Calculate cancellation rate by month
    monthly_cancellation = df.groupby('arrival_date_month')['is_canceled'].agg(['count', 'sum']).reset_index()
    monthly_cancellation['cancellation_rate'] = (monthly_cancellation['sum'] / monthly_cancellation['count']) * 100
    monthly_cancellation.columns = ['month', 'total_bookings', 'canceled_bookings', 'cancellation_rate']
    
    # Sort by month order
    monthly_cancellation['month_order'] = monthly_cancellation['month'].apply(lambda x: month_order.index(x))
    monthly_cancellation = monthly_cancellation.sort_values('month_order')
    monthly_cancellation = monthly_cancellation.drop('month_order', axis=1)
    
    return monthly_cancellation

def plot_cancellation_by_month(monthly_cancellation):
    """Plot cancellation rates by month"""
    plt.figure(figsize=(14, 8))
    sns.barplot(x='month', y='cancellation_rate', data=monthly_cancellation)
    plt.title('Cancellation Rate by Month')
    plt.xlabel('Month')
    plt.ylabel('Cancellation Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cancellation_by_month.png'))
    plt.close()

def analyze_cancellation_by_market_segment(df):
    """Analyze cancellation rates by market segment"""
    market_cancellation = df.groupby('market_segment')['is_canceled'].agg(['count', 'sum']).reset_index()
    market_cancellation['cancellation_rate'] = (market_cancellation['sum'] / market_cancellation['count']) * 100
    market_cancellation.columns = ['market_segment', 'total_bookings', 'canceled_bookings', 'cancellation_rate']
    market_cancellation = market_cancellation.sort_values('cancellation_rate', ascending=False)
    
    return market_cancellation

def plot_cancellation_by_market_segment(market_cancellation):
    """Plot cancellation rates by market segment"""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='market_segment', y='cancellation_rate', data=market_cancellation)
    plt.title('Cancellation Rate by Market Segment')
    plt.xlabel('Market Segment')
    plt.ylabel('Cancellation Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cancellation_by_market_segment.png'))
    plt.close()

def main():
    """Main function to run the cancellation rate analysis"""
    # Load data
    df = load_data()
    
    # Calculate overall cancellation rate
    overall_rate = calculate_overall_cancellation_rate(df)
    
    # Analyze cancellation by hotel type
    hotel_cancellation = analyze_cancellation_by_hotel_type(df)
    plot_cancellation_by_hotel_type(hotel_cancellation)
    
    # Analyze cancellation by month
    monthly_cancellation = analyze_cancellation_by_month(df)
    plot_cancellation_by_month(monthly_cancellation)
    
    # Analyze cancellation by market segment
    market_cancellation = analyze_cancellation_by_market_segment(df)
    plot_cancellation_by_market_segment(market_cancellation)
    
    return {
        'overall_rate': overall_rate,
        'hotel_cancellation': hotel_cancellation.to_dict('records'),
        'monthly_cancellation': monthly_cancellation.to_dict('records'),
        'market_cancellation': market_cancellation.to_dict('records')
    }

if __name__ == "__main__":
    main()