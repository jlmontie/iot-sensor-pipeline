import numpy as np
import pandas as pd

# Removed unused plotly imports
from datetime import datetime, timedelta
from scipy.optimize import curve_fit, minimize_scalar
from sklearn.ensemble import RandomForestRegressor

# Removed unused sklearn.metrics import
import warnings

warnings.filterwarnings("ignore")


class SmartWateringPredictor:
    """
    Advanced watering predictor with automatic fallback methods.

    Uses machine learning as primary method, with automatic fallback to:
    1. ML Random Forest (primary) - for datasets with 50+ points
    2. Exponential decay modeling (fallback) - for smaller datasets
    3. Linear regression (final fallback) - when curve fitting fails

    Provides comprehensive plant health analysis and watering predictions.
    """

    def __init__(self, watering_threshold=0.3, critical_threshold=0.2):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.watering_threshold = watering_threshold
        self.critical_threshold = critical_threshold
        self.confidence_score = 0.0
        self.prediction_method = "ml"  # Track which method was used
        self.last_analysis = None

    def analyze(self, df):
        """
        Analyzes moisture data using the best available prediction method.

        Automatically selects prediction method based on data quality:
        1. ML Random Forest (if sufficient data and training succeeds)
        2. Exponential decay (if curve fitting succeeds)
        3. Linear regression (final fallback)

        Parameters:
        -----------
        df : pandas.DataFrame
            Must contain columns: 'timestamp' (datetime) and 'moisture' (float 0-1)

        Returns:
        --------
        dict : Complete analysis results with prediction method used
        """
        # Validate and prepare data
        df_prepared = self._validate_and_prepare(df)

        # Detect watering events
        watering_event = self._detect_watering_event(df_prepared)

        # Get predictions using best available method
        prediction = self._predict_next_watering(df_prepared)

        # Assess plant health
        health = self._assess_plant_health(df_prepared, watering_event, prediction)

        # Generate recommendations
        recommendations = self._generate_recommendations(health, prediction)

        # Compile final results
        return self._compile_results(
            watering_event, prediction, health, recommendations, df_prepared
        )

    def _validate_and_prepare(self, df):
        """Validate input and prepare data for analysis."""
        self._validate_input(df)
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)

        # Determine best prediction method based on data quality
        self._select_prediction_method(df_sorted)

        return df_sorted

    def _select_prediction_method(self, df):
        """Select the best prediction method based on data availability."""
        data_points = len(df)

        if data_points >= 50:
            # Try ML training
            try:
                self._auto_train(df)
                if self.is_trained and self.confidence_score > 0.3:
                    self.prediction_method = "ml"
                    return
            except Exception:
                pass

        # Fall back to exponential decay
        self.prediction_method = "exponential"

    def _detect_watering_event(self, df):
        """Detect last watering event using pattern recognition."""
        return self._detect_last_watering_event(df)

    def _predict_next_watering(self, df):
        """Predict next watering using the selected method."""
        if self.prediction_method == "ml":
            return self._predict_next_watering_ml(df)
        elif self.prediction_method == "exponential":
            return self._predict_next_watering_exponential(df)
        else:
            return self._predict_next_watering_linear(df)

    def _assess_plant_health(self, df, watering_event, prediction):
        """Calculate comprehensive plant health metrics."""
        return self._calculate_health_metrics(df, watering_event, prediction)

    def _compile_results(self, watering_event, prediction, health, recommendations, df):
        """Compile all analysis results into final output."""
        # Store analysis metadata
        self.last_analysis = {
            "timestamp": datetime.now(),
            "data_points": len(df),
            "analysis_period": (df["timestamp"].min(), df["timestamp"].max()),
            "prediction_method": self.prediction_method,
        }

        return {
            "last_watering_event": watering_event,
            "next_watering_prediction": prediction,
            "moisture_decay_curve": self._generate_decay_curve(df, prediction),
            "plant_health_metrics": health,
            "recommendations": recommendations,
            "model_confidence": self.confidence_score,
            "prediction_method_used": self.prediction_method,
            "analysis_metadata": self.last_analysis,
        }

    def _validate_input(self, df):
        """Validate input DataFrame structure and data quality"""
        required_columns = ["timestamp", "moisture"]

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        if len(df) < 10:
            raise ValueError(
                "Insufficient data points. Need at least 10 readings for analysis."
            )

        # Check for NaN values
        if df[required_columns].isnull().any().any():
            raise ValueError("Data contains NaN values. Please clean the data first.")

        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            raise ValueError("'timestamp' column must be datetime type")

        # Validate moisture range
        if not df["moisture"].between(0, 1).all():
            raise ValueError("'moisture' values must be between 0 and 1")

    def _auto_train(self, df):
        """Automatically train the model using the provided data"""
        print("Auto-training SmartWateringPredictor...")

        # Use first 70% for training
        split_idx = int(len(df) * 0.7)
        train_df = df[:split_idx].copy()

        X_train = []
        y_train = []

        # Create training features
        for i in range(24, len(train_df)):  # Need at least 24 hours of history
            moisture_history = train_df["moisture"].iloc[i - 24 : i].values
            current_time = train_df["timestamp"].iloc[i]

            # Create time features
            time_features = {
                "hour_of_day": current_time.hour,
                "day_of_week": current_time.weekday(),
                "season": (current_time.month % 12) // 3,
                "days_since_last_water": self._estimate_days_since_watering(
                    moisture_history
                ),
            }

            # Create features
            features = self._create_features(moisture_history, time_features)
            X_train.append(features.flatten())
            y_train.append(train_df["moisture"].iloc[i])

        if len(X_train) > 0:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.confidence_score = 0.85  # Base confidence after training
            print(f"Training completed with {len(X_train)} samples")
        else:
            print("Insufficient data for training")
            self.confidence_score = 0.3

    def _estimate_days_since_watering(self, moisture_history):
        """Estimate days since last watering from moisture history"""
        if len(moisture_history) < 2:
            return 0

        # Look for significant moisture increases
        for i in range(len(moisture_history) - 1, 0, -1):
            if moisture_history[i] - moisture_history[i - 1] > 0.2:
                return (len(moisture_history) - i) / 24
        return len(moisture_history) / 24

    def _detect_last_watering_event(self, df):
        """Machine learning watering event detection"""
        # Calculate moisture differences
        df["moisture_diff"] = df["moisture"].diff()
        df["time_diff_hours"] = df["timestamp"].diff().dt.total_seconds() / 3600

        # Find significant moisture increases
        watering_candidates = df[
            (df["moisture_diff"] > 0.15)  # Significant increase
            & (df["time_diff_hours"] < 6)  # Within reasonable time gap
            & (df["moisture"].shift(1) < 0.7)  # Previous moisture wasn't already high
        ].copy()

        if len(watering_candidates) > 0:
            # Get the most recent watering event
            last_watering_row = watering_candidates.iloc[-1]

            return {
                "detected": True,
                "timestamp": last_watering_row["timestamp"],
                "confidence": min(0.95, 0.5 + last_watering_row["moisture_diff"] * 2),
                "moisture_before": df.loc[last_watering_row.name - 1, "moisture"],
                "moisture_after": last_watering_row["moisture"],
                "increase_amount": last_watering_row["moisture_diff"],
                "method": "Machine learning pattern recognition",
            }
        else:
            # Estimate based on highest moisture point
            max_moisture_idx = df["moisture"].idxmax()
            days_since = (
                df["timestamp"].max() - df.loc[max_moisture_idx, "timestamp"]
            ).total_seconds() / (24 * 3600)

            return {
                "detected": False,
                "message": f"No clear watering events detected. Estimated {days_since:.1f} days since last watering.",
                "estimated_days_since_watering": days_since,
                "confidence": 0.4,
                "method": "ML-based estimation",
            }

    def _predict_next_watering_ml(self, df):
        """ML-based watering prediction with extended horizon"""
        current_time = df["timestamp"].max()
        current_moisture = df["moisture"].iloc[-1]

        # Check immediate watering needs
        if current_moisture <= self.critical_threshold:
            return {
                "timestamp": current_time,
                "hours_from_now": 0,
                "predicted_moisture": current_moisture,
                "urgency": "CRITICAL",
                "confidence": 0.95,
                "message": "Immediate watering required - critically dry!",
            }
        elif current_moisture <= self.watering_threshold:
            return {
                "timestamp": current_time,
                "hours_from_now": 0,
                "predicted_moisture": current_moisture,
                "urgency": "HIGH",
                "confidence": 0.9,
                "message": "Watering recommended now",
            }

        # Use ML model to predict future moisture levels
        moisture_history = df["moisture"].tail(24).values

        # Predict up to 336 hours (14 days) to find watering point
        for hours_ahead in range(1, 337):
            future_time = current_time + timedelta(hours=hours_ahead)

            # Create time features for prediction
            time_features = {
                "hour_of_day": future_time.hour,
                "day_of_week": future_time.weekday(),
                "season": (future_time.month % 12) // 3,
                "days_since_last_water": self._estimate_days_since_watering(
                    moisture_history
                )
                + hours_ahead / 24,
            }

            # Create features and predict
            features = self._create_features(moisture_history, time_features)
            predicted_moisture = self.model.predict(features)[0]
            predicted_moisture = max(0, min(1, predicted_moisture))  # Clamp to [0,1]

            # Check if watering threshold is reached
            if predicted_moisture <= self.watering_threshold:
                # Determine urgency
                if hours_ahead < 12:
                    urgency = "HIGH"
                elif hours_ahead < 48:
                    urgency = "MEDIUM"
                else:
                    urgency = "LOW"

                return {
                    "timestamp": future_time,
                    "hours_from_now": float(hours_ahead),
                    "predicted_moisture": predicted_moisture,
                    "urgency": urgency,
                    "confidence": self.confidence_score,
                    "message": f"ML prediction based on {len(moisture_history)}h history",
                }

            # Update moisture history for next prediction (simple decay simulation)
            if len(moisture_history) >= 24:
                moisture_history = np.append(moisture_history[1:], predicted_moisture)
            else:
                moisture_history = np.append(moisture_history, predicted_moisture)

        # If no watering needed in 14 days
        return {
            "timestamp": current_time + timedelta(hours=336),
            "hours_from_now": 336.0,
            "predicted_moisture": predicted_moisture,
            "urgency": "NONE",
            "confidence": 0.6,
            "message": "No watering needed in next 14 days (ML prediction)",
        }

    def _predict_next_watering_exponential(self, df):
        """Predict next watering using exponential decay modeling."""
        current_time = df["timestamp"].max()
        current_moisture = df["moisture"].iloc[-1]

        # Check immediate watering needs
        if current_moisture <= self.critical_threshold:
            return {
                "timestamp": current_time,
                "hours_from_now": 0,
                "predicted_moisture": current_moisture,
                "urgency": "CRITICAL",
                "confidence": 0.99,
                "message": "Plant needs immediate watering - critically dry!",
            }

        if current_moisture <= self.watering_threshold:
            return {
                "timestamp": current_time,
                "hours_from_now": 0,
                "predicted_moisture": current_moisture,
                "urgency": "HIGH",
                "confidence": 0.95,
                "message": "Plant needs watering now",
            }

        # Try exponential decay model
        try:
            decay_model = self._fit_exponential_decay_model(df)
            if decay_model["fitted"]:
                return self._predict_with_decay_model(df, decay_model)
        except Exception:
            pass

        # Fall back to linear if exponential fails
        return self._predict_next_watering_linear(df)

    def _predict_next_watering_linear(self, df):
        """Final fallback: simple linear regression prediction."""
        current_time = df["timestamp"].max()
        current_moisture = df["moisture"].iloc[-1]

        # Check immediate watering needs
        if current_moisture <= self.critical_threshold:
            return {
                "timestamp": current_time,
                "hours_from_now": 0,
                "predicted_moisture": current_moisture,
                "urgency": "CRITICAL",
                "confidence": 0.99,
                "message": "Plant needs immediate watering - critically dry!",
            }

        if current_moisture <= self.watering_threshold:
            return {
                "timestamp": current_time,
                "hours_from_now": 0,
                "predicted_moisture": current_moisture,
                "urgency": "HIGH",
                "confidence": 0.95,
                "message": "Plant needs watering now",
            }

        # Simple linear decay estimation
        recent_data = df.tail(min(48, len(df)))  # Use last 48 hours or available data
        if len(recent_data) >= 2:
            # Calculate decay rate from recent trend
            time_span_hours = (
                recent_data["timestamp"].max() - recent_data["timestamp"].min()
            ).total_seconds() / 3600
            moisture_change = (
                recent_data["moisture"].iloc[0] - recent_data["moisture"].iloc[-1]
            )
            decay_rate = max(
                0.001, moisture_change / max(time_span_hours, 1)
            )  # Avoid division by zero

            # Calculate hours until watering threshold
            hours_to_threshold = (
                current_moisture - self.watering_threshold
            ) / decay_rate
            hours_to_threshold = max(
                1, min(336, hours_to_threshold)
            )  # Clamp between 1 hour and 14 days

            # Set confidence and urgency
            if hours_to_threshold < 12:
                urgency = "HIGH"
                confidence = 0.7
            elif hours_to_threshold < 48:
                urgency = "MEDIUM"
                confidence = 0.6
            else:
                urgency = "LOW"
                confidence = 0.5

            self.confidence_score = confidence

            return {
                "timestamp": current_time + timedelta(hours=int(hours_to_threshold)),
                "hours_from_now": float(hours_to_threshold),
                "predicted_moisture": self.watering_threshold,
                "urgency": urgency,
                "confidence": confidence,
                "message": f"Linear trend prediction based on {len(recent_data)} recent readings",
            }

        # Ultimate fallback: default estimate
        default_hours = 72  # 3 days default
        self.confidence_score = 0.3

        return {
            "timestamp": current_time + timedelta(hours=default_hours),
            "hours_from_now": float(default_hours),
            "predicted_moisture": self.watering_threshold,
            "urgency": "MEDIUM",
            "confidence": 0.3,
            "message": "Default estimate - insufficient data for reliable prediction",
        }

    def _fit_exponential_decay_model(self, df):
        """Fit exponential decay model to moisture data."""
        # Use data from the last 7 days or all available data
        recent_data = df.tail(min(168, len(df))).copy()

        if len(recent_data) < 10:
            return {
                "fitted": False,
                "message": "Insufficient data for exponential modeling",
            }

        # Convert timestamps to hours since start
        start_time = recent_data["timestamp"].min()
        recent_data["hours_since_start"] = (
            recent_data["timestamp"] - start_time
        ).dt.total_seconds() / 3600

        # Define exponential decay function: moisture = A * exp(-t/tau) + offset
        def exponential_decay(t, A, tau, offset):
            return A * np.exp(-t / tau) + offset

        try:
            x_data = recent_data["hours_since_start"].values
            y_data = recent_data["moisture"].values

            # Initial parameter guesses
            A_guess = max(0.1, y_data[0] - y_data[-1])
            tau_guess = len(x_data) * 2
            offset_guess = max(0.05, min(y_data))

            # Fit with bounds
            popt, pcov = curve_fit(
                exponential_decay,
                x_data,
                y_data,
                p0=[A_guess, tau_guess, offset_guess],
                bounds=([0, 10, 0], [2, 500, 1]),
                maxfev=2000,
            )

            # Calculate model quality
            y_pred = exponential_decay(x_data, *popt)
            r_squared = 1 - np.sum((y_data - y_pred) ** 2) / np.sum(
                (y_data - np.mean(y_data)) ** 2
            )

            # Set confidence based on fit quality
            self.confidence_score = max(0.3, min(0.95, r_squared))

            return {
                "fitted": True,
                "parameters": {"amplitude": popt[0], "tau": popt[1], "offset": popt[2]},
                "quality_metrics": {
                    "r_squared": r_squared,
                    "confidence": self.confidence_score,
                },
                "model_function": lambda t: exponential_decay(t, *popt),
                "start_time": start_time,
            }

        except Exception as e:
            return {"fitted": False, "message": f"Exponential fit failed: {str(e)}"}

    def _predict_with_decay_model(self, df, decay_model):
        """Make prediction using fitted exponential decay model."""
        current_time = df["timestamp"].max()

        model_func = decay_model["model_function"]
        start_time = decay_model["start_time"]
        hours_since_start = (current_time - start_time).total_seconds() / 3600

        # Find when moisture will reach watering threshold
        try:

            def moisture_at_time(hours_ahead):
                return model_func(hours_since_start + hours_ahead)

            result = minimize_scalar(
                lambda h: abs(moisture_at_time(h) - self.watering_threshold),
                bounds=(0, 336),
                method="bounded",
            )

            hours_ahead = float(result.x)
            predicted_moisture = moisture_at_time(hours_ahead)

            # Determine urgency
            if hours_ahead < 12:
                urgency = "HIGH"
            elif hours_ahead < 48:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"

            return {
                "timestamp": current_time + timedelta(hours=int(hours_ahead)),
                "hours_from_now": hours_ahead,
                "predicted_moisture": predicted_moisture,
                "urgency": urgency,
                "confidence": self.confidence_score,
                "message": f'Exponential decay prediction (R² = {decay_model["quality_metrics"]["r_squared"]:.3f})',
            }

        except Exception:
            # Fall back to linear if optimization fails
            return self._predict_next_watering_linear(df)

    def _generate_decay_curve(self, df, prediction):
        """Generate decay curve using the selected prediction method."""
        if self.prediction_method == "ml":
            return self._generate_decay_curve_ml(df, hours_ahead=336)
        elif self.prediction_method == "exponential":
            return self._generate_decay_curve_exponential(df, hours_ahead=336)
        else:
            return self._generate_decay_curve_linear(df, hours_ahead=336)

    def _generate_decay_curve_ml(self, df, hours_ahead=336):
        """Generate ML-based moisture decay curve"""
        current_time = df["timestamp"].max()
        current_moisture = df["moisture"].iloc[-1]
        moisture_history = df["moisture"].tail(24).values

        curve_data = []

        for h in range(hours_ahead + 1):
            future_time = current_time + timedelta(hours=h)

            if h == 0:
                predicted_moisture = current_moisture
            else:
                # Create time features
                time_features = {
                    "hour_of_day": future_time.hour,
                    "day_of_week": future_time.weekday(),
                    "season": (future_time.month % 12) // 3,
                    "days_since_last_water": self._estimate_days_since_watering(
                        moisture_history
                    )
                    + h / 24,
                }

                # Predict moisture
                features = self._create_features(moisture_history, time_features)
                predicted_moisture = self.model.predict(features)[0]
                predicted_moisture = max(0, min(1, predicted_moisture))

                # Update history for next prediction
                if len(moisture_history) >= 24:
                    moisture_history = np.append(
                        moisture_history[1:], predicted_moisture
                    )
                else:
                    moisture_history = np.append(moisture_history, predicted_moisture)

            # Determine status and color
            if predicted_moisture <= self.critical_threshold:
                status = "CRITICAL"
                color = "red"
            elif predicted_moisture <= self.watering_threshold:
                status = "NEEDS_WATERING"
                color = "orange"
            else:
                status = "HEALTHY"
                color = "green"

            # Calculate time-decaying confidence
            confidence = max(0.3, self.confidence_score * (1 - h / (hours_ahead * 2)))

            curve_data.append(
                {
                    "timestamp": future_time,
                    "predicted_moisture": predicted_moisture,
                    "status": status,
                    "color": color,
                    "confidence": confidence,
                    "hours_from_now": h,
                }
            )

        return {
            "curve_data": curve_data,
            "prediction_horizon_hours": hours_ahead,
            "model_type": "ml_random_forest",
            "base_confidence": self.confidence_score,
        }

    def _generate_decay_curve_exponential(self, df, hours_ahead=336):
        """Generate exponential decay curve."""
        current_time = df["timestamp"].max()

        # Try to fit exponential model
        decay_model = self._fit_exponential_decay_model(df)
        curve_data = []

        if decay_model["fitted"]:
            model_func = decay_model["model_function"]
            start_time = decay_model["start_time"]
            hours_since_start = (current_time - start_time).total_seconds() / 3600

            for h in range(hours_ahead + 1):
                future_time = current_time + timedelta(hours=h)
                predicted_moisture = max(0, min(1, model_func(hours_since_start + h)))

                # Determine status
                if predicted_moisture <= self.critical_threshold:
                    status, color = "CRITICAL", "red"
                elif predicted_moisture <= self.watering_threshold:
                    status, color = "NEEDS_WATERING", "orange"
                else:
                    status, color = "HEALTHY", "green"

                confidence = max(
                    0.3, self.confidence_score * (1 - h / (hours_ahead * 2))
                )

                curve_data.append(
                    {
                        "timestamp": future_time,
                        "predicted_moisture": predicted_moisture,
                        "status": status,
                        "color": color,
                        "confidence": confidence,
                        "hours_from_now": h,
                    }
                )

            model_type = "exponential_decay"
        else:
            # Fall back to linear if exponential fails
            return self._generate_decay_curve_linear(df, hours_ahead)

        return {
            "curve_data": curve_data,
            "prediction_horizon_hours": hours_ahead,
            "model_type": model_type,
            "base_confidence": self.confidence_score,
        }

    def _generate_decay_curve_linear(self, df, hours_ahead=336):
        """Generate linear decay curve as final fallback."""
        current_time = df["timestamp"].max()
        current_moisture = df["moisture"].iloc[-1]

        # Calculate simple linear decay rate
        recent_data = df.tail(min(48, len(df)))
        if len(recent_data) >= 2:
            time_span_hours = (
                recent_data["timestamp"].max() - recent_data["timestamp"].min()
            ).total_seconds() / 3600
            moisture_change = (
                recent_data["moisture"].iloc[0] - recent_data["moisture"].iloc[-1]
            )
            decay_rate = max(0.001, moisture_change / max(time_span_hours, 1))
        else:
            decay_rate = 0.005  # Default decay rate

        curve_data = []
        for h in range(hours_ahead + 1):
            future_time = current_time + timedelta(hours=h)
            predicted_moisture = max(0, current_moisture - (h * decay_rate))

            # Determine status
            if predicted_moisture <= self.critical_threshold:
                status, color = "CRITICAL", "red"
            elif predicted_moisture <= self.watering_threshold:
                status, color = "NEEDS_WATERING", "orange"
            else:
                status, color = "HEALTHY", "green"

            confidence = max(0.2, 0.6 * (1 - h / (hours_ahead * 3)))

            curve_data.append(
                {
                    "timestamp": future_time,
                    "predicted_moisture": predicted_moisture,
                    "status": status,
                    "color": color,
                    "confidence": confidence,
                    "hours_from_now": h,
                }
            )

        return {
            "curve_data": curve_data,
            "prediction_horizon_hours": hours_ahead,
            "model_type": "linear_fallback",
            "base_confidence": 0.5,
        }

    def _calculate_health_metrics(self, df, last_watering, next_watering):
        """Calculate comprehensive plant health metrics"""
        current_moisture = df["moisture"].iloc[-1]

        # Calculate 7-day average
        seven_days_ago = df["timestamp"].max() - timedelta(days=7)
        recent_data = df[df["timestamp"] >= seven_days_ago]
        avg_moisture_7d = recent_data["moisture"].mean()

        # Calculate health score (0-100)
        health_score = min(
            100,
            max(
                0,
                (
                    current_moisture * 40  # Current moisture weight
                    + avg_moisture_7d * 30  # Recent average weight
                    + (1 - abs(current_moisture - 0.5)) * 30  # Optimal range bonus
                )
                * 100,
            ),
        )

        # Days since last watering
        if last_watering["detected"]:
            days_since_watering = (
                df["timestamp"].max() - last_watering["timestamp"]
            ).total_seconds() / (24 * 3600)
        else:
            days_since_watering = last_watering.get("estimated_days_since_watering", 0)

        return {
            "health_score": health_score,
            "current_moisture": current_moisture,
            "average_moisture_7d": avg_moisture_7d,
            "days_since_last_watering": days_since_watering,
            "moisture_stability": recent_data["moisture"].std(),
            "time_in_optimal_range": len(
                recent_data[recent_data["moisture"].between(0.3, 0.7)]
            )
            / len(recent_data),
            "watering_frequency_analysis": self._analyze_watering_frequency(df),
        }

    def _generate_recommendations(self, health_metrics, next_watering):
        """Generate smart recommendations based on analysis"""
        recommendations = []

        if health_metrics["health_score"] >= 90:
            recommendations.append(
                "Excellent plant health! Continue current care routine."
            )
        elif health_metrics["health_score"] >= 70:
            recommendations.append("Good plant health with room for optimization.")
        else:
            recommendations.append(
                "Plant health needs attention. Review watering schedule."
            )

        if next_watering["urgency"] == "CRITICAL":
            recommendations.append("URGENT: Water immediately to prevent plant stress!")
        elif next_watering["urgency"] == "HIGH":
            recommendations.append("Water soon to maintain optimal moisture levels.")
        elif next_watering["urgency"] == "MEDIUM":
            recommendations.append("Plan to water in the next day or two.")

        if health_metrics["moisture_stability"] > 0.15:
            recommendations.append(
                "High moisture variability detected. Consider more consistent watering."
            )

        return recommendations

    def _analyze_watering_frequency(self, df):
        """Analyze watering frequency patterns"""
        # Simplified frequency analysis
        return {
            "average_days_between_watering": 7,  # Placeholder
            "consistency_score": 0.8,  # Placeholder
            "recommendation": "Maintain current schedule",
        }

    def _create_features(
        self, moisture_history, time_features, environmental_data=None
    ):
        """Create features for ML prediction"""
        features = []

        # Time-based features
        features.extend(
            [
                time_features["hour_of_day"],
                time_features["day_of_week"],
                time_features["season"],
                time_features["days_since_last_water"],
            ]
        )

        # Moisture pattern features
        if len(moisture_history) >= 24:
            features.extend(
                [
                    np.mean(moisture_history[-24:]),  # 24h average
                    np.std(moisture_history[-24:]),  # 24h volatility
                    np.min(moisture_history[-24:]),  # 24h minimum
                    (moisture_history[-1] - moisture_history[-24]) / 24,  # decay rate
                    len(
                        [
                            i
                            for i in range(1, len(moisture_history))
                            if moisture_history[i] - moisture_history[i - 1] > 0.3
                        ]
                    ),  # watering events
                ]
            )
        else:
            features.extend([0, 0, 0, 0, 0])

        # Environmental features (if available)
        if environmental_data:
            features.extend(
                [
                    environmental_data.get("temperature", 20),
                    environmental_data.get("humidity", 50),
                    environmental_data.get("light_hours", 12),
                    environmental_data.get("season_factor", 1.0),
                ]
            )
        else:
            features.extend([20, 50, 12, 1.0])

        return np.array(features).reshape(1, -1)
