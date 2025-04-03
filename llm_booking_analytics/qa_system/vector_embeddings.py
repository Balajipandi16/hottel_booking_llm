# Code for vector embedding storage
import pandas as pd
import numpy as np
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModel
import weaviate
from weaviate.embedded import EmbeddedOptions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class VectorEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the vector embeddings class"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.client = None
        self.class_name = "HotelBooking"
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, text):
        """Get embedding for a text"""
        # Tokenize and compute embeddings
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Mean pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings[0].numpy()
    
    def setup_weaviate(self):
        """Setup Weaviate client and schema"""
        # Initialize Weaviate client
        try:
            # Try with embedded options
            self.client = weaviate.Client(
                embedded_options=EmbeddedOptions()
            )
        except:
            # Fallback to simple embedded mode
            self.client = weaviate.Client(embedded=True)
        
        # Define schema if it doesn't exist
        if not self.client.schema.exists(self.class_name):
            class_obj = {
                "class": self.class_name,
                "description": "Hotel booking data with vector embeddings",
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {"name": "hotel", "dataType": ["string"]},
                    {"name": "is_canceled", "dataType": ["boolean"]},
                    {"name": "lead_time", "dataType": ["int"]},
                    {"name": "arrival_date", "dataType": ["string"]},
                    {"name": "stays_in_weekend_nights", "dataType": ["int"]},
                    {"name": "stays_in_week_nights", "dataType": ["int"]},
                    {"name": "adults", "dataType": ["int"]},
                    {"name": "children", "dataType": ["int"]},
                    {"name": "babies", "dataType": ["int"]},
                    {"name": "meal", "dataType": ["string"]},
                    {"name": "country", "dataType": ["string"]},
                    {"name": "market_segment", "dataType": ["string"]},
                    {"name": "reserved_room_type", "dataType": ["string"]},
                    {"name": "assigned_room_type", "dataType": ["string"]},
                    {"name": "adr", "dataType": ["number"]},
                    {"name": "content", "dataType": ["text"]}
                ]
            }
            self.client.schema.create_class(class_obj)
            
    def load_and_index_data(self, sample_size=1000):
        """Load and index hotel booking data"""
        # Setup Weaviate
        logging.info("Setting up Weaviate client...")
        self.setup_weaviate()
        
        # Clear existing data
        logging.info("Clearing existing data...")
        try:
            self.client.schema.delete_all()
            logging.info("Successfully cleared existing data")
        except Exception as e:
            logging.error(f"Error clearing data: {e}")
        
        self.setup_weaviate()
        
        # Load data
        logging.info("Loading data...")
        data_path = os.path.join('data', 'hotel_bookings (1).csv')
        
        if not os.path.exists(data_path):
            logging.error(f"Data file not found: {data_path}")
            # Check if file exists in current directory
            current_dir = os.getcwd()
            logging.info(f"Current directory: {current_dir}")
            files = os.listdir(current_dir)
            logging.info(f"Files in current directory: {files}")
            
            # Try to find the data file in the data directory
            if os.path.exists('data'):
                data_files = os.listdir('data')
                logging.info(f"Files in data directory: {data_files}")
                
                # Try alternative filenames
                alternative_paths = [
                    os.path.join('data', 'hotel_bookings.csv'),
                    os.path.join('data', 'hotel_bookings (1).csv'),
                    os.path.join('data', 'hotel_bookings(1).csv')
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        data_path = alt_path
                        logging.info(f"Found alternative data file: {data_path}")
                        break
        
        try:
            df = pd.read_csv(data_path)
            logging.info(f"Successfully loaded data with {len(df)} rows")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return
        
        # Take a sample for indexing (for efficiency)
        if sample_size > 0 and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            logging.info(f"Sampled {len(df)} rows from data")
        
        # Process and index each row in batches
        logging.info(f"Indexing {len(df)} hotel bookings...")
        batch_size = 50
        with self.client.batch as batch:
            for i, row in df.iterrows():
                try:
                    # Create a text representation of the booking
                    text_content = f"Hotel: {row['hotel']}. "
                    text_content += f"Canceled: {'Yes' if row['is_canceled'] == 1 else 'No'}. "
                    text_content += f"Lead time: {row['lead_time']} days. "
                    text_content += f"Arrival date: {row['arrival_date_month']} {row['arrival_date_day_of_month']}, {row['arrival_date_year']}. "
                    text_content += f"Stay duration: {row['stays_in_weekend_nights']} weekend nights, {row['stays_in_week_nights']} weekday nights. "
                    text_content += f"Guests: {row['adults']} adults, {row['children']} children, {row['babies']} babies. "
                    text_content += f"Meal plan: {row['meal']}. Country: {row['country']}. "
                    text_content += f"Market segment: {row['market_segment']}. "
                    text_content += f"Room types: reserved {row['reserved_room_type']}, assigned {row['assigned_room_type']}. "
                    text_content += f"ADR: {row['adr']}. "
                except Exception as e:
                    logging.error(f"Error creating text content for row {i}: {e}")
                    continue
            
                try:
                    # Get embedding
                    logging.debug(f"Getting embedding for row {i}")
                    embedding = self.get_embedding(text_content)
                    
                    # Prepare data object
                    data_object = {
                        "hotel": str(row['hotel']),
                        "is_canceled": bool(row['is_canceled']),
                        "lead_time": int(row['lead_time']),
                        "arrival_date": f"{row['arrival_date_month']} {row['arrival_date_day_of_month']}, {row['arrival_date_year']}",
                        "stays_in_weekend_nights": int(row['stays_in_weekend_nights']),
                        "stays_in_week_nights": int(row['stays_in_week_nights']),
                        "adults": int(row['adults']),
                        "children": float(row['children']) if not pd.isna(row['children']) else 0,
                        "babies": int(row['babies']),
                        "meal": str(row['meal']),
                        "country": str(row['country']),
                        "market_segment": str(row['market_segment']),
                        "reserved_room_type": str(row['reserved_room_type']),
                        "assigned_room_type": str(row['assigned_room_type']),
                        "adr": float(row['adr']),
                        "content": text_content
                    }
                except Exception as e:
                    logging.error(f"Error preparing data for row {i}: {e}")
                    continue
                
                try:
                    # Add to batch
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=self.class_name,
                        vector=embedding.tolist()
                    )
                    
                    if i % batch_size == 0 and i > 0:
                        logging.info(f"Indexed {i} bookings...")
                except Exception as e:
                    logging.error(f"Error adding data object to batch for row {i}: {e}")
                    continue
        
        logging.info("Indexing complete!")
        
        # Verify data was indexed
        try:
            count = self.client.query.aggregate(self.class_name).with_meta_count().do()
            if 'data' in count and 'Aggregate' in count['data'] and self.class_name in count['data']['Aggregate']:
                count_value = count['data']['Aggregate'][self.class_name][0]['meta']['count']
                logging.info(f"Successfully indexed {count_value} objects")
            else:
                logging.warning("Could not verify indexed count")
        except Exception as e:
            logging.error(f"Error verifying indexed count: {e}")
            
    def semantic_search(self, query, limit=5):
        """Perform semantic search on the indexed data"""
        logging.info(f"Performing semantic search for query: {query}")
        
        if self.client is None:
            logging.info("Weaviate client not initialized, setting up...")
            try:
                self.setup_weaviate()
                logging.info("Weaviate client setup successful")
            except Exception as e:
                logging.error(f"Error setting up Weaviate: {e}")
                return []
        
        # Verify that we have data in the index
        try:
            count = self.client.query.aggregate(self.class_name).with_meta_count().do()
            if 'data' in count and 'Aggregate' in count['data'] and self.class_name in count['data']['Aggregate']:
                count_value = count['data']['Aggregate'][self.class_name][0]['meta']['count']
                logging.info(f"Found {count_value} objects in index")
                if count_value == 0:
                    logging.warning("No data in index, attempting to load data...")
                    self.load_and_index_data(sample_size=1000)
            else:
                logging.warning("Could not verify indexed count, attempting to load data...")
                self.load_and_index_data(sample_size=1000)
        except Exception as e:
            logging.error(f"Error checking index count: {e}")
            # Try to reload data
            self.load_and_index_data(sample_size=1000)
            
        # Get query embedding
        try:
            logging.info("Getting query embedding...")
            query_embedding = self.get_embedding(query)
            logging.info("Successfully got query embedding")
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            return []
        
        try:
            # Perform vector search with timeout handling
            import socket
            socket.setdefaulttimeout(30)  # Set a 30-second timeout for socket operations
            
            # Perform vector search
            logging.info("Executing vector search...")
            result = self.client.query.get(
                self.class_name, ["hotel", "is_canceled", "lead_time", "arrival_date", "content"]
            ).with_near_vector({
                "vector": query_embedding.tolist()
            }).with_limit(limit).do()
            
            logging.info(f"Search result: {result}")
            
            if 'data' in result and 'Get' in result['data'] and self.class_name in result['data']['Get']:
                results = result["data"]["Get"][self.class_name]
                logging.info(f"Found {len(results)} results")
                return results
            else:
                logging.warning("No results found in search response")
                logging.info(f"Response structure: {result.keys() if result else 'None'}")
                return []
        except ConnectionResetError as e:
            logging.error(f"Connection reset error in semantic search: {e}")
            # Try to reconnect
            try:
                self.client = None
                self.setup_weaviate()
                logging.info("Reconnected to Weaviate after connection reset")
            except Exception as reconnect_error:
                logging.error(f"Failed to reconnect: {reconnect_error}")
            return []
        except Exception as e:
            logging.error(f"Error in semantic search: {e}")
            return []

def main():
    """Main function to demonstrate vector embeddings"""
    # Initialize vector embeddings
    vector_embeddings = VectorEmbeddings()
    
    # Load and index data (sample for demonstration)
    vector_embeddings.load_and_index_data(sample_size=1000)
    
    # Perform semantic search
    query = "bookings with long lead times that were canceled"
    results = vector_embeddings.semantic_search(query)
    
    # Print results
    print(f"Query: {query}")
    print("Results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Hotel: {result['hotel']}")
        print(f"Canceled: {result['is_canceled']}")
        print(f"Lead time: {result['lead_time']}")
        print(f"Arrival date: {result['arrival_date']}")
        print(f"Content: {result['content'][:200]}...")
        print()

if __name__ == "__main__":
    main()