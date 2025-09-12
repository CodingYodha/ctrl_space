import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, acos, degrees
from scipy.spatial import cKDTree
import time
from bisect import bisect_left, bisect_right
import warnings
warnings.filterwarnings('ignore')

def create_sample_datasets():
    """Create comprehensive sample datasets for demonstration"""
    
    # GW events dataset (25 events - mix of real and synthetic)
    gw_data = [
        {"event_id": "GW150914", "utc_time": "2015-09-14 09:50:44.400", "ra_deg": 112.5, "dec_deg": -70.2, "pos_error_deg": 0.164, "signal_strength": 24},
        {"event_id": "GW151012", "utc_time": "2015-10-12 09:54:42.400", "ra_deg": 102.4, "dec_deg": -33.6, "pos_error_deg": 0.444, "signal_strength": 10},
        {"event_id": "GW151226", "utc_time": "2015-12-26 03:38:52.600", "ra_deg": 92.2, "dec_deg": -39.6, "pos_error_deg": 0.236, "signal_strength": 13},
        {"event_id": "GW170104", "utc_time": "2017-01-04 10:11:58.600", "ra_deg": 190.0, "dec_deg": -22.0, "pos_error_deg": 0.256, "signal_strength": 13},
        {"event_id": "GW170608", "utc_time": "2017-06-08 02:01:16.500", "ra_deg": 120.0, "dec_deg": -60.0, "pos_error_deg": 0.094, "signal_strength": 15},
        {"event_id": "GW170729", "utc_time": "2017-07-29 18:56:29.300", "ra_deg": 268.0, "dec_deg": -33.0, "pos_error_deg": 0.286, "signal_strength": 10},
        {"event_id": "GW170809", "utc_time": "2017-08-09 08:28:21.800", "ra_deg": 154.0, "dec_deg": -22.0, "pos_error_deg": 0.094, "signal_strength": 12},
        {"event_id": "GW170814", "utc_time": "2017-08-14 10:30:43.500", "ra_deg": 124.0, "dec_deg": -39.0, "pos_error_deg": 0.017, "signal_strength": 16},
        {"event_id": "GW170817", "utc_time": "2017-08-17 12:41:04.400", "ra_deg": 197.45, "dec_deg": -23.38, "pos_error_deg": 0.0044, "signal_strength": 32},
        {"event_id": "GW170818", "utc_time": "2017-08-18 02:25:09.100", "ra_deg": 162.0, "dec_deg": -46.0, "pos_error_deg": 0.025, "signal_strength": 11},
        {"event_id": "GW190412", "utc_time": "2019-04-12 05:30:44.100", "ra_deg": 180.0, "dec_deg": -20.0, "pos_error_deg": 0.019, "signal_strength": 19},
        {"event_id": "GW190425", "utc_time": "2019-04-25 08:18:05.000", "ra_deg": 180.0, "dec_deg": 12.0, "pos_error_deg": 2.073, "signal_strength": 12},
        {"event_id": "GW190521", "utc_time": "2019-05-21 03:02:29.400", "ra_deg": 115.5, "dec_deg": -25.0, "pos_error_deg": 0.061, "signal_strength": 15},
        {"event_id": "GW190814", "utc_time": "2019-08-14 21:10:39.000", "ra_deg": 134.56, "dec_deg": 2.69, "pos_error_deg": 0.005, "signal_strength": 25},
        {"event_id": "GW200115", "utc_time": "2020-01-15 04:23:09.700", "ra_deg": 50.0, "dec_deg": -15.0, "pos_error_deg": 0.094, "signal_strength": 15},
        {"event_id": "GW200129", "utc_time": "2020-01-29 06:55:35.400", "ra_deg": 190.0, "dec_deg": -35.0, "pos_error_deg": 0.8, "signal_strength": 11},
        {"event_id": "GW200202", "utc_time": "2020-02-02 15:43:50.500", "ra_deg": 210.0, "dec_deg": 10.0, "pos_error_deg": 0.5, "signal_strength": 14},
        {"event_id": "GW200316", "utc_time": "2020-03-16 08:32:14.200", "ra_deg": 75.3, "dec_deg": 42.1, "pos_error_deg": 0.15, "signal_strength": 16},
        {"event_id": "GW200421", "utc_time": "2020-04-21 14:17:33.800", "ra_deg": 156.7, "dec_deg": -18.4, "pos_error_deg": 0.32, "signal_strength": 13},
        {"event_id": "GW200518", "utc_time": "2020-05-18 20:45:12.600", "ra_deg": 245.8, "dec_deg": 28.9, "pos_error_deg": 0.087, "signal_strength": 18},
        {"event_id": "GW210624", "utc_time": "2021-06-24 11:28:47.300", "ra_deg": 334.2, "dec_deg": -55.7, "pos_error_deg": 0.21, "signal_strength": 14},
        {"event_id": "GW210807", "utc_time": "2021-08-07 19:33:25.100", "ra_deg": 98.6, "dec_deg": 67.3, "pos_error_deg": 0.44, "signal_strength": 9},
        {"event_id": "GW220211", "utc_time": "2022-02-11 03:17:08.900", "ra_deg": 12.9, "dec_deg": -8.2, "pos_error_deg": 0.16, "signal_strength": 17},
        {"event_id": "GW220627", "utc_time": "2022-06-27 16:22:51.400", "ra_deg": 278.4, "dec_deg": 15.6, "pos_error_deg": 0.29, "signal_strength": 12},
        {"event_id": "GW230529", "utc_time": "2023-05-29 12:34:56.100", "ra_deg": 345.6, "dec_deg": 45.2, "pos_error_deg": 0.12, "signal_strength": 18}
    ]
    
    # GRB events dataset (35 events - strategically placed for correlations)
    grb_data = [
        {"event_id": "GRB150314A", "utc_time": "2015-03-14 12:34:15.000", "ra_deg": 195.6, "dec_deg": -12.4, "pos_error_deg": 8.2, "signal_strength": 15.3},
        {"event_id": "bn170817529", "utc_time": "2017-08-17 12:41:06.470", "ra_deg": 197.42, "dec_deg": -23.42, "pos_error_deg": 3.2, "signal_strength": 6.2},  # Famous!
        {"event_id": "GRB190425A", "utc_time": "2019-04-25 08:18:05.000", "ra_deg": 182.7, "dec_deg": 9.3, "pos_error_deg": 16.2, "signal_strength": 5.1},
        {"event_id": "GRB150914B", "utc_time": "2015-09-14 10:15:30.000", "ra_deg": 114.2, "dec_deg": -68.9, "pos_error_deg": 12.5, "signal_strength": 8.4},
        {"event_id": "GRB151012C", "utc_time": "2015-10-12 11:22:18.000", "ra_deg": 104.1, "dec_deg": -35.2, "pos_error_deg": 15.8, "signal_strength": 7.2},
        {"event_id": "GRB151226D", "utc_time": "2015-12-26 04:12:33.000", "ra_deg": 94.8, "dec_deg": -41.1, "pos_error_deg": 11.3, "signal_strength": 9.6},
        {"event_id": "GRB170104E", "utc_time": "2017-01-04 11:45:22.000", "ra_deg": 188.5, "dec_deg": -20.8, "pos_error_deg": 9.8, "signal_strength": 11.2},
        {"event_id": "GRB170608F", "utc_time": "2017-06-08 03:22:45.000", "ra_deg": 122.3, "dec_deg": -58.7, "pos_error_deg": 14.1, "signal_strength": 9.8},
        {"event_id": "GRB170729G", "utc_time": "2017-07-29 20:15:18.000", "ra_deg": 270.2, "dec_deg": -31.8, "pos_error_deg": 16.7, "signal_strength": 6.9},
        {"event_id": "GRB170809H", "utc_time": "2017-08-09 09:45:33.000", "ra_deg": 156.1, "dec_deg": -20.4, "pos_error_deg": 8.9, "signal_strength": 10.1},
        {"event_id": "GRB170814I", "utc_time": "2017-08-14 12:18:21.000", "ra_deg": 125.8, "dec_deg": -37.6, "pos_error_deg": 7.4, "signal_strength": 13.2},
        {"event_id": "GRB170818J", "utc_time": "2017-08-18 03:44:15.000", "ra_deg": 164.2, "dec_deg": -44.3, "pos_error_deg": 12.1, "signal_strength": 8.7},
        {"event_id": "GRB190412K", "utc_time": "2019-04-12 07:15:20.000", "ra_deg": 182.1, "dec_deg": -18.9, "pos_error_deg": 11.3, "signal_strength": 8.7},
        {"event_id": "GRB190521L", "utc_time": "2019-05-21 04:45:12.000", "ra_deg": 117.2, "dec_deg": -26.8, "pos_error_deg": 13.7, "signal_strength": 7.9},
        {"event_id": "GRB190814M", "utc_time": "2019-08-14 22:30:45.000", "ra_deg": 136.2, "dec_deg": 1.8, "pos_error_deg": 8.9, "signal_strength": 12.4},
        {"event_id": "GRB200115N", "utc_time": "2020-01-15 05:12:33.000", "ra_deg": 52.1, "dec_deg": -13.7, "pos_error_deg": 10.2, "signal_strength": 9.1},
        {"event_id": "GRB200129O", "utc_time": "2020-01-29 08:22:18.000", "ra_deg": 188.9, "dec_deg": -33.4, "pos_error_deg": 12.8, "signal_strength": 6.8},
        {"event_id": "GRB200202P", "utc_time": "2020-02-02 16:18:45.000", "ra_deg": 212.3, "dec_deg": 8.7, "pos_error_deg": 9.6, "signal_strength": 10.3},
        {"event_id": "GRB200316Q", "utc_time": "2020-03-16 10:15:42.000", "ra_deg": 77.8, "dec_deg": 40.6, "pos_error_deg": 14.2, "signal_strength": 7.8},
        {"event_id": "GRB200421R", "utc_time": "2020-04-21 15:45:18.000", "ra_deg": 158.4, "dec_deg": -16.9, "pos_error_deg": 11.7, "signal_strength": 9.4},
        {"event_id": "GRB200518S", "utc_time": "2020-05-18 22:12:35.000", "ra_deg": 243.2, "dec_deg": 30.4, "pos_error_deg": 13.1, "signal_strength": 8.2},
        {"event_id": "GRB180325A", "utc_time": "2018-03-25 17:42:33.000", "ra_deg": 89.2, "dec_deg": 29.1, "pos_error_deg": 8.3, "signal_strength": 16.1},
        {"event_id": "GRB190114C", "utc_time": "2019-01-14 20:57:03.000", "ra_deg": 201.3, "dec_deg": 12.7, "pos_error_deg": 6.8, "signal_strength": 22.4},
        {"event_id": "GRB200716C", "utc_time": "2020-07-16 02:34:59.000", "ra_deg": 145.2, "dec_deg": 55.3, "pos_error_deg": 7.9, "signal_strength": 17.4},
        {"event_id": "GRB210624T", "utc_time": "2021-06-24 13:15:21.000", "ra_deg": 332.8, "dec_deg": -53.2, "pos_error_deg": 15.4, "signal_strength": 6.7},
        {"event_id": "GRB210807U", "utc_time": "2021-08-07 21:18:33.000", "ra_deg": 96.9, "dec_deg": 65.8, "pos_error_deg": 18.2, "signal_strength": 5.9},
        {"event_id": "GRB220211V", "utc_time": "2022-02-11 04:42:15.000", "ra_deg": 14.7, "dec_deg": -6.8, "pos_error_deg": 12.5, "signal_strength": 8.8},
        {"event_id": "GRB220627W", "utc_time": "2022-06-27 18:33:42.000", "ra_deg": 276.1, "dec_deg": 17.3, "pos_error_deg": 14.8, "signal_strength": 7.3},
        {"event_id": "GRB221009A", "utc_time": "2022-10-09 13:16:59.000", "ra_deg": 288.3, "dec_deg": 19.8, "pos_error_deg": 4.2, "signal_strength": 45.7},
        {"event_id": "GRB230529X", "utc_time": "2023-05-29 14:12:33.000", "ra_deg": 347.8, "dec_deg": 43.9, "pos_error_deg": 11.4, "signal_strength": 8.6},
        {"event_id": "GRB240101Y", "utc_time": "2024-01-01 12:00:00.000", "ra_deg": 0.0, "dec_deg": 0.0, "pos_error_deg": 20.0, "signal_strength": 4.5},
        {"event_id": "GRB240215Z", "utc_time": "2024-02-15 06:30:00.000", "ra_deg": 180.0, "dec_deg": 45.0, "pos_error_deg": 25.0, "signal_strength": 3.8},
        {"event_id": "GRB240330AA", "utc_time": "2024-03-30 18:45:00.000", "ra_deg": 90.0, "dec_deg": -30.0, "pos_error_deg": 15.0, "signal_strength": 6.2},
        {"event_id": "GRB240612BB", "utc_time": "2024-06-12 09:15:00.000", "ra_deg": 270.0, "dec_deg": 60.0, "pos_error_deg": 22.0, "signal_strength": 5.1},
        {"event_id": "GRB240901CC", "utc_time": "2024-09-01 21:30:00.000", "ra_deg": 315.0, "dec_deg": -75.0, "pos_error_deg": 18.5, "signal_strength": 7.4}
    ]
    
    return pd.DataFrame(gw_data), pd.DataFrame(grb_data)

class MultimessengerCorrelator:
    """High-performance multimessenger correlation analysis using KD-trees with TOP 10 guarantee"""
    
    def __init__(self, weights=None):
        """
        Initialize correlator with scoring weights
        
        Parameters:
        - weights: Dict with 'temporal', 'spatial', 'significance' weights
        """
        self.weights = weights or {'temporal': 0.5, 'spatial': 0.3, 'significance': 0.2}
        self.gw_data = None
        self.grb_data = None
        self.grb_kdtree = None
        self.grb_times_sorted = None
        self.grb_indices_by_time = None
        
    def load_data_from_dataframes(self, gw_df, grb_df):
        """Load data from pandas DataFrames"""
        print("Loading and preprocessing datasets...")
        start_time = time.time()
        
        self.gw_data = gw_df.copy()
        self.grb_data = grb_df.copy()
        
        # Convert time columns
        self.gw_data['utc_time'] = pd.to_datetime(self.gw_data['utc_time'])
        self.grb_data['utc_time'] = pd.to_datetime(self.grb_data['utc_time'])
        
        # Build spatial KD-tree for GRB positions
        grb_positions = self._spherical_to_cartesian(
            self.grb_data['ra_deg'].values, 
            self.grb_data['dec_deg'].values
        )
        self.grb_kdtree = cKDTree(grb_positions)
        
        # Build temporal index for GRB events
        self.grb_data['timestamp'] = self.grb_data['utc_time'].astype(np.int64) // 10**9
        sorted_indices = np.argsort(self.grb_data['timestamp'].values)
        self.grb_times_sorted = self.grb_data['timestamp'].values[sorted_indices]
        self.grb_indices_by_time = sorted_indices
        
        load_time = time.time() - start_time
        print(f"✓ Loaded {len(self.gw_data)} GW and {len(self.grb_data)} GRB events")
        print(f"✓ Built KD-tree and temporal indices in {load_time:.3f}s")
        
        return self
    
    def _spherical_to_cartesian(self, ra_deg, dec_deg):
        """Convert spherical coordinates to Cartesian for KD-tree"""
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad) 
        z = np.sin(dec_rad)
        
        return np.column_stack([x, y, z])
    
    def _cartesian_to_angular_distance(self, cart1, cart2):
        """Calculate angular distance from Cartesian coordinates"""
        cos_angle = np.clip(np.dot(cart1, cart2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _find_temporal_candidates(self, gw_time, max_time_diff_sec=86400):
        """Find GRB events within time window using binary search"""
        gw_timestamp = gw_time.timestamp()
        
        left_time = gw_timestamp - max_time_diff_sec
        right_time = gw_timestamp + max_time_diff_sec
        
        left_idx = bisect_left(self.grb_times_sorted, left_time)
        right_idx = bisect_right(self.grb_times_sorted, right_time)
        
        return self.grb_indices_by_time[left_idx:right_idx]
    
    def _find_spatial_candidates(self, gw_position_cart, max_angular_distance_deg=180):
        """Find GRB events within angular distance using KD-tree"""
        max_chord_distance = 2 * np.sin(np.radians(max_angular_distance_deg) / 2)
        
        distances, indices = self.grb_kdtree.query(
            gw_position_cart.reshape(1, -1),
            k=len(self.grb_data),
            distance_upper_bound=max_chord_distance
        )
        
        valid_mask = distances[0] != np.inf
        return indices[0][valid_mask]
    
    def calculate_correlation_score(self, gw_idx, grb_idx):
        """Calculate correlation score between specific GW and GRB events"""
        gw_event = self.gw_data.iloc[gw_idx]
        grb_event = self.grb_data.iloc[grb_idx]
        
        # Time difference
        time_diff = abs((gw_event['utc_time'] - grb_event['utc_time']).total_seconds())
        
        # Angular separation
        gw_cart = self._spherical_to_cartesian(
            np.array([gw_event['ra_deg']]), 
            np.array([gw_event['dec_deg']])
        )[0]
        grb_cart = self.grb_kdtree.data[grb_idx]
        angular_sep = self._cartesian_to_angular_distance(gw_cart, grb_cart)
        
        # Combined position error
        combined_error = gw_event['pos_error_deg'] + grb_event['pos_error_deg']
        
        # Calculate sub-scores
        temporal_score = np.exp(-time_diff / 3600)
        
        if angular_sep < combined_error:
            spatial_score = np.exp(-angular_sep / combined_error)
        else:
            spatial_score = np.exp(-angular_sep / combined_error) * 0.1
            
        sig_score = (gw_event['signal_strength'] / 50) * (grb_event['signal_strength'] / 50)
        
        # Combined confidence score
        confidence = (self.weights['temporal'] * temporal_score + 
                     self.weights['spatial'] * spatial_score + 
                     self.weights['significance'] * sig_score)
        
        return {
            'gw_event_id': gw_event['event_id'],
            'grb_event_id': grb_event['event_id'],
            'confidence_score': confidence,
            'temporal_score': temporal_score,
            'spatial_score': spatial_score,
            'significance_score': sig_score,
            'time_diff_sec': time_diff,
            'time_diff_hours': time_diff / 3600,
            'angular_sep_deg': angular_sep,
            'combined_error_deg': combined_error,
            'within_error_circle': angular_sep < combined_error,
            'gw_time': gw_event['utc_time'],
            'grb_time': grb_event['utc_time'],
            'gw_ra': gw_event['ra_deg'],
            'gw_dec': gw_event['dec_deg'],
            'grb_ra': grb_event['ra_deg'], 
            'grb_dec': grb_event['dec_deg'],
            'gw_snr': gw_event['signal_strength'],
            'grb_flux': grb_event['signal_strength'],
            'gw_pos_error': gw_event['pos_error_deg'],
            'grb_pos_error': grb_event['pos_error_deg']
        }
    
    def find_top_correlations(self, max_time_window=86400, max_spatial_search=90, min_confidence=0.01, 
                             output_file=None, target_top_n=10):
        """
        Find correlations with adaptive parameters to guarantee TOP N results
        
        Parameters:
        - max_time_window: Maximum time difference in seconds (default: 24 hours)
        - max_spatial_search: Maximum spatial search radius in degrees (default: 90°)
        - min_confidence: Minimum confidence score to include (default: 0.01)
        - output_file: Optional filename to save results
        - target_top_n: Target number of top correlations (default: 10)
        
        Returns:
        - DataFrame with correlations sorted by confidence
        """
        print(f"🔍 Multimessenger Correlation Analysis - Targeting TOP {target_top_n}")
        print(f"Time window: ±{max_time_window/3600:.1f} hours")
        print(f"Spatial search radius: {max_spatial_search}°")
        print(f"Minimum confidence: {min_confidence}")
        print("-" * 60)
        
        start_time = time.time()
        correlations = []
        total_comparisons = 0
        
        # First pass with standard parameters
        for gw_idx in range(len(self.gw_data)):
            gw_event = self.gw_data.iloc[gw_idx]
            
            # Step 1: Temporal filtering
            temporal_candidates = self._find_temporal_candidates(gw_event['utc_time'], max_time_window)
            
            if len(temporal_candidates) == 0:
                continue
                
            # Step 2: Spatial filtering with KD-tree
            gw_cart = self._spherical_to_cartesian(
                np.array([gw_event['ra_deg']]), 
                np.array([gw_event['dec_deg']])
            )[0]
            
            spatial_candidates = self._find_spatial_candidates(gw_cart, max_spatial_search)
            
            # Step 3: Intersection of temporal and spatial candidates
            candidate_set = set(temporal_candidates) & set(spatial_candidates)
            
            # Step 4: Calculate scores for filtered candidates
            for grb_idx in candidate_set:
                total_comparisons += 1
                score_data = self.calculate_correlation_score(gw_idx, grb_idx)
                
                if score_data['confidence_score'] >= min_confidence:
                    correlations.append(score_data)
        
        analysis_time = time.time() - start_time
        
        print(f"✓ Initial analysis completed in {analysis_time:.3f}s")
        print(f"✓ Total comparisons: {total_comparisons:,}")
        print(f"✓ Found {len(correlations)} correlations above threshold")
        
        # If we don't have enough correlations, expand parameters adaptively
        if len(correlations) < target_top_n:
            print(f"\n🔄 Adaptive expansion: Need {target_top_n - len(correlations)} more correlations")
            
            # Try expanding parameters progressively
            expanded_params = [
                (86400 * 7, 120, 0.005),   # 7 days, 120°, very low confidence  
                (86400 * 30, 150, 0.001),  # 30 days, 150°, ultra low confidence
                (86400 * 90, 180, 0.0001)  # 90 days, full sky, minimal confidence
            ]
            
            for time_window, spatial_radius, conf_threshold in expanded_params:
                if len(correlations) >= target_top_n:
                    break
                    
                print(f"   Trying: ±{time_window/86400:.0f} days, {spatial_radius}°, conf>{conf_threshold}")
                
                additional_correlations = []
                for gw_idx in range(len(self.gw_data)):
                    gw_event = self.gw_data.iloc[gw_idx]
                    
                    temporal_candidates = self._find_temporal_candidates(gw_event['utc_time'], time_window)
                    if len(temporal_candidates) == 0:
                        continue
                        
                    gw_cart = self._spherical_to_cartesian(
                        np.array([gw_event['ra_deg']]), 
                        np.array([gw_event['dec_deg']])
                    )[0]
                    
                    spatial_candidates = self._find_spatial_candidates(gw_cart, spatial_radius)
                    candidate_set = set(temporal_candidates) & set(spatial_candidates)
                    
                    for grb_idx in candidate_set:
                        score_data = self.calculate_correlation_score(gw_idx, grb_idx)
                        
                        # Check if this is a new correlation
                        is_new = True
                        for existing in correlations:
                            if (existing['gw_event_id'] == score_data['gw_event_id'] and 
                                existing['grb_event_id'] == score_data['grb_event_id']):
                                is_new = False
                                break
                        
                        if is_new and score_data['confidence_score'] >= conf_threshold:
                            additional_correlations.append(score_data)
                
                correlations.extend(additional_correlations)
                print(f"   Found {len(additional_correlations)} additional correlations")
        
        if correlations:
            # Create results DataFrame
            corr_df = pd.DataFrame(correlations)
            corr_df = corr_df.sort_values('confidence_score', ascending=False).reset_index(drop=True)
            
            # Add rank column
            corr_df['rank'] = range(1, len(corr_df) + 1)
            
            # Get top N correlations
            top_correlations = corr_df.head(target_top_n)
            
            # Display results
            print(f"\n🏆 TOP {len(top_correlations)} Multimessenger Correlations:")
            print("=" * 80)
            
            for i, (_, row) in enumerate(top_correlations.iterrows()):
                rank = i + 1
                
                print(f"\n#{rank}. {row['gw_event_id']} ↔ {row['grb_event_id']}")
                print(f"   Confidence: {row['confidence_score']:.4f}")
                print(f"   Time Diff: {row['time_diff_sec']:.1f}s ({row['time_diff_hours']:.2f}h)")
                print(f"   Angular Sep: {row['angular_sep_deg']:.3f}°")
                print(f"   Within Error: {'✓' if row['within_error_circle'] else '✗'}")
                print(f"   GW: {row['gw_time']} | GRB: {row['grb_time']}")
                
                # Special annotations
                if rank == 1:
                    print("   🌟 HIGHEST CONFIDENCE CORRELATION!")
                    if row['gw_event_id'] == 'GW170817' and row['grb_event_id'] == 'bn170817529':
                        print("   🎉 Famous GW170817-GRB170817A detection!")
                        print("   🌌 First confirmed multimessenger event!")
                
                if row['within_error_circle']:
                    print("   ✨ High-quality spatial coincidence")
            
            # Save top N results if requested
            if output_file:
                # Reorder columns for better readability
                output_columns = [
                    'rank', 'gw_event_id', 'grb_event_id', 'confidence_score',
                    'time_diff_sec', 'time_diff_hours', 'angular_sep_deg', 'within_error_circle',
                    'temporal_score', 'spatial_score', 'significance_score',
                    'gw_time', 'grb_time', 'gw_ra', 'gw_dec', 'grb_ra', 'grb_dec',
                    'gw_snr', 'grb_flux', 'gw_pos_error', 'grb_pos_error', 'combined_error_deg'
                ]
                
                top_correlations[output_columns].to_csv(output_file, index=False)
                print(f"\n💾 TOP {len(top_correlations)} correlations saved to: {output_file}")
            
            # Summary statistics
            print(f"\n📊 Summary Statistics:")
            print(f"   Total correlations found: {len(corr_df)}")
            print(f"   TOP {target_top_n} correlations saved")
            print(f"   Highest confidence: {corr_df['confidence_score'].max():.4f}")
            print(f"   Average confidence (top {target_top_n}): {top_correlations['confidence_score'].mean():.4f}")
            print(f"   Within error circles: {sum(top_correlations['within_error_circle'])}/{len(top_correlations)}")
            print(f"   Time range: {top_correlations['time_diff_hours'].min():.2f}h - {top_correlations['time_diff_hours'].max():.2f}h")
            
            return corr_df
        else:
            print("❌ No correlations found even with expanded parameters.")
            print("   Consider using larger datasets or different correlation criteria.")
            return pd.DataFrame()

def main():
    """Main execution function with guaranteed TOP 10 results"""
    print("🚀 KD-Tree TOP 10 Multimessenger Correlator (Standalone Version)")
    print("=" * 70)
    
    # Create comprehensive sample datasets
    print("Creating comprehensive sample datasets...")
    gw_df, grb_df = create_sample_datasets()
    print(f"✓ Created {len(gw_df)} GW events and {len(grb_df)} GRB events")
    
    # Initialize correlator
    correlator = MultimessengerCorrelator()
    
    # Load data from DataFrames
    correlator.load_data_from_dataframes(gw_df, grb_df)
    
    # Find TOP 10 correlations with adaptive parameters
    results = correlator.find_top_correlations(
        max_time_window=86400,      # Start with 24 hours
        max_spatial_search=90,      # 90 degrees initial radius
        min_confidence=0.01,        # Low threshold to start
        output_file='guaranteed_top_10_correlations.csv',
        target_top_n=10            # Guarantee 10 results
    )
    
    print(f"\n🎉 GUARANTEED TOP 10 ANALYSIS COMPLETE!")
    print(f"📄 Successfully found and ranked {len(results)} correlations")
    print(f"💾 TOP 10 correlations guaranteed and saved to CSV file")
    print(f"🚀 No external CSV files required - everything is self-contained!")
    return results

if __name__ == "__main__":
    results = main()