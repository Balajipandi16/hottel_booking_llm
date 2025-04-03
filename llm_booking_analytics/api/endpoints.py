# API endpoints
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, List, Optional
import pandas as pd
import os
from pydantic import BaseModel

# Import analytics modules
from analytics.distribution import analyze_country_distribution
from analytics.cancellation_rate import calculate_overall_cancellation_rate, analyze_cancellation_by_hotel_type, analyze_cancellation_by_month, analyze_cancellation_by_market_segment
from analytics.booking_lead_time import calculate_lead_time_statistics, analyze_lead_time_by_hotel_type, analyze_lead_time_by_market_segment, analyze_lead_time_cancellation_correlation
from analytics.additional_analytics import analyze_stay_duration, analyze_adr, analyze_special_requests, analyze_room_assignment
from qa_system.bert_qa import BertQA

router = APIRouter()

# Load data once at module level with memory-efficient settings
try:
    data_path = os.path.join('data', 'hotel_bookings (1).csv')
    if not os.path.exists(data_path):
        # Try alternative paths
        alternative_paths = [
            os.path.join('data', 'hotel_bookings.csv'),
            os.path.join('data', 'hotel_bookings(1).csv'),
            os.path.join('.', 'data', 'hotel_bookings (1).csv')
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                break
    
    # Use a smaller chunk size and only load necessary columns to reduce memory usage
    df = pd.read_csv(
        data_path,
        dtype={
            'is_canceled': 'int8',
            'lead_time': 'int32',
            'stays_in_weekend_nights': 'int8',
            'stays_in_week_nights': 'int8',
            'adults': 'int8',
            'children': 'float32',
            'babies': 'int8'
        }
    )
    print(f"Successfully loaded dataset with {len(df)} rows")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Create an empty DataFrame with the expected columns as a fallback
    df = pd.DataFrame(columns=['hotel', 'is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_month', 
                              'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
                              'adults', 'children', 'babies', 'meal', 'country', 'market_segment',
                              'reserved_room_type', 'assigned_room_type', 'adr'])

@router.get("/data/summary", response_model=Dict)
async def get_data_summary():
    """Get a summary of the dataset"""
    try:
        summary = {
            "total_records": len(df),
            "columns": list(df.columns),
            "hotel_types": df['hotel'].unique().tolist(),
            "date_range": f"{df['arrival_date_year'].min()}-{df['arrival_date_year'].max()}",
            "countries": len(df['country'].unique()),
        }
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distribution/countries", response_model=Dict)
async def get_country_distribution(top_n: int = Query(15, description="Number of top countries to return")):
    """Get the geographical distribution of hotel bookings"""
    try:
        # Count bookings by country
        country_counts = df['country'].value_counts().reset_index()
        country_counts.columns = ['country', 'count']
        
        # Get top N countries
        top_countries = country_counts.head(top_n).to_dict('records')
        
        return {"top_countries": top_countries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cancellation/overall", response_model=Dict)
async def get_overall_cancellation_rate():
    """Get the overall cancellation rate"""
    try:
        overall_rate = calculate_overall_cancellation_rate(df)
        return overall_rate
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cancellation/by-hotel", response_model=Dict)
async def get_cancellation_by_hotel_type():
    """Get cancellation rates by hotel type"""
    try:
        hotel_cancellation = analyze_cancellation_by_hotel_type(df)
        return {"hotel_cancellation": hotel_cancellation.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cancellation/by-month", response_model=Dict)
async def get_cancellation_by_month():
    """Get cancellation rates by month"""
    try:
        monthly_cancellation = analyze_cancellation_by_month(df)
        return {"monthly_cancellation": monthly_cancellation.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cancellation/by-market-segment", response_model=Dict)
async def get_cancellation_by_market_segment():
    """Get cancellation rates by market segment"""
    try:
        market_cancellation = analyze_cancellation_by_market_segment(df)
        return {"market_cancellation": market_cancellation.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lead-time/statistics", response_model=Dict)
async def get_lead_time_statistics():
    """Get basic statistics for booking lead time"""
    try:
        lead_time_stats = calculate_lead_time_statistics(df)
        return lead_time_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lead-time/by-hotel", response_model=Dict)
async def get_lead_time_by_hotel_type():
    """Get lead time statistics by hotel type"""
    try:
        hotel_lead_time = analyze_lead_time_by_hotel_type(df)
        return {"hotel_lead_time": hotel_lead_time.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lead-time/by-market-segment", response_model=Dict)
async def get_lead_time_by_market_segment():
    """Get lead time statistics by market segment"""
    try:
        market_lead_time = analyze_lead_time_by_market_segment(df)
        return {"market_lead_time": market_lead_time.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lead-time/cancellation-correlation", response_model=Dict)
async def get_lead_time_cancellation_correlation():
    """Get correlation between lead time and cancellation rate"""
    try:
        lead_time_cancellation = analyze_lead_time_cancellation_correlation(df)
        return {"lead_time_cancellation": lead_time_cancellation.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stay-duration/statistics", response_model=Dict)
async def get_stay_duration_statistics():
    """Get statistics for stay duration"""
    try:
        stay_stats, hotel_stay = analyze_stay_duration(df)
        return {
            "stay_stats": stay_stats,
            "hotel_stay": hotel_stay.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/adr/statistics", response_model=Dict)
async def get_adr_statistics():
    """Get statistics for Average Daily Rate (ADR)"""
    try:
        adr_stats, hotel_adr = analyze_adr(df)
        return {
            "adr_stats": adr_stats,
            "hotel_adr": hotel_adr.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/special-requests/statistics", response_model=Dict)
async def get_special_requests_statistics():
    """Get statistics for special requests"""
    try:
        special_requests_stats, special_requests_counts = analyze_special_requests(df)
        return {
            "special_requests_stats": special_requests_stats,
            "special_requests_counts": special_requests_counts.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/room-assignment/statistics", response_model=Dict)
async def get_room_assignment_statistics():
    """Get statistics for room assignment vs. reservation"""
    try:
        match_rate, hotel_match_rate = analyze_room_assignment(df)
        return {
            "overall_match_rate": match_rate,
            "hotel_match_rate": hotel_match_rate.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# QA System models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class InsightRequest(BaseModel):
    sample_size: Optional[int] = 1000

# Initialize QA system
qa_system = None

@router.post("/qa/answer", response_model=Dict)
async def answer_question(request: QuestionRequest):
    """Answer a question about hotel bookings using the BERT QA system"""
    global qa_system
    
    try:
        # Initialize QA system if not already initialized
        if qa_system is None:
            try:
                print("Initializing QA system...")
                qa_system = BertQA()
                # Use a smaller sample size to avoid memory issues
                qa_system.load_and_prepare_data(sample_size=500)
                print("QA system initialized successfully")
            except MemoryError as mem_error:
                print(f"Memory error initializing QA system: {mem_error}")
                return {
                    "answer": "The system is currently experiencing memory constraints. Please try again later with a simpler question.",
                    "error": "Memory allocation error",
                    "score": 0.0,
                    "context": "",
                    "sources": []
                }
            except Exception as init_error:
                print(f"Error initializing QA system: {init_error}")
                return {
                    "answer": "Unable to initialize the QA system. Please try again later.",
                    "error": str(init_error),
                    "score": 0.0,
                    "context": "",
                    "sources": []
                }
        
        # Get answer from QA system with timeout handling
        try:
            print(f"Processing question: {request.question}")
            # Limit top_k to avoid memory issues
            safe_top_k = min(request.top_k, 3)
            answer = qa_system.answer_question(request.question, top_k=safe_top_k)
            print("Successfully processed question")
            return answer
        except ConnectionResetError as conn_error:
            print(f"Connection reset error: {conn_error}")
            return {
                "answer": "Connection was reset while processing your question. Please try again later.",
                "error": "ECONNRESET",
                "score": 0.0,
                "context": "",
                "sources": []
            }
        except MemoryError as mem_error:
            print(f"Memory error processing question: {mem_error}")
            return {
                "answer": "The system is currently experiencing memory constraints. Please try again with a simpler question.",
                "error": "Memory allocation error",
                "score": 0.0,
                "context": "",
                "sources": []
            }
        except Exception as qa_error:
            print(f"Error processing question: {qa_error}")
            return {
                "answer": "An error occurred while processing your question.",
                "error": str(qa_error),
                "score": 0.0,
                "context": "",
                "sources": []
            }
    except Exception as e:
        return {
            "answer": "An unexpected error occurred.",
            "error": str(e),
            "score": 0.0,
            "context": "",
            "sources": []
        }

@router.post("/qa/insights", response_model=Dict)
async def generate_insights(request: InsightRequest = Body(...)):
    """Generate insights about hotel bookings using the BERT QA system"""
    global qa_system
    
    try:
        # Initialize QA system if not already initialized
        if qa_system is None:
            try:
                print("Initializing QA system for insights...")
                qa_system = BertQA()
                # Limit sample size to avoid memory issues
                safe_sample_size = min(request.sample_size, 500)
                qa_system.load_and_prepare_data(sample_size=safe_sample_size)
                print("QA system initialized successfully for insights")
            except MemoryError as mem_error:
                print(f"Memory error initializing QA system for insights: {mem_error}")
                return {
                    "insights": ["The system is currently experiencing memory constraints. Please try again later with a smaller sample size."],
                    "error": "Memory allocation error"
                }
            except Exception as init_error:
                print(f"Error initializing QA system for insights: {init_error}")
                return {
                    "insights": [f"Unable to initialize the QA system: {str(init_error)}"],
                    "error": str(init_error)
                }
        
        try:
            # Use a sample of the data for insights with a safe limit
            safe_sample_size = min(request.sample_size, 500, len(df))
            print(f"Generating insights with sample size: {safe_sample_size}")
            
            try:
                sample_df = df.sample(safe_sample_size)
            except Exception as sample_error:
                print(f"Error sampling data: {sample_error}")
                # If sampling fails, use a smaller sample or the first few rows
                if len(df) > 0:
                    sample_df = df.head(min(100, len(df)))
                else:
                    return {
                        "insights": ["No data available to generate insights."],
                        "error": "No data available"
                    }
            
            # Generate insights
            insights = qa_system.generate_analytics_insights(sample_df)
            print("Successfully generated insights")
            return {"insights": insights}
        except ConnectionResetError as conn_error:
            print(f"Connection reset error generating insights: {conn_error}")
            return {
                "insights": ["Connection was reset while generating insights. Please try again later."],
                "error": "ECONNRESET"
            }
        except MemoryError as mem_error:
            print(f"Memory error generating insights: {mem_error}")
            return {
                "insights": ["The system is currently experiencing memory constraints. Please try again with a smaller sample size."],
                "error": "Memory allocation error"
            }
        except Exception as insights_error:
            print(f"Error generating insights: {insights_error}")
            return {
                "insights": [f"An error occurred while generating insights: {str(insights_error)}"],
                "error": str(insights_error)
            }
    except Exception as e:
        return {
            "insights": [f"An unexpected error occurred: {str(e)}"],
            "error": str(e)
        }