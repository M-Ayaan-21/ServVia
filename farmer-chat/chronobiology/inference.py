"""
ServVia Chronobiology Inference Engine
=======================================

Lightweight module that estimates a user's biological state from minimal
contextual inputs — local time, IP-derived geolocation, and live weather data.
Requires **NO wearable data** and makes **ZERO LLM calls**.

External APIs used (both free, no API key required):
    - ip-api.com        — IP-to-geolocation (lat/lon, timezone, city, country)
    - open-meteo.com    — Real-time weather (temperature, weather code)

Theoretical Foundations:
    - **Two-Process Model** (Borbély, 1982): Process C (circadian) and
      Process S (homeostatic sleep pressure).
    - **Ayurvedic Dinacharya**: Traditional Indian daily routine mapped
      to dosha dominance periods (Vata/Pitta/Kapha).
    - **Ritucharya**: Ayurvedic seasonal regimen linking climate to
      physiological tendencies.
    - **Seasonal Affective Context**: Overcast/rainy weather documented
      to increase subjective fatigue and elevate homeostatic sleep drive.

Author: ServVia Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import requests

from core.models import (
    BiologicalState,
    CircadianPhase,
    SeasonalInfluence,
    SleepPressure,
)

logger = logging.getLogger("ServVia.ChronobiologyEngine")


# =============================================================================
# WEATHER CODE MAPPING (WMO Weather Code Standard — used by Open-Meteo)
# =============================================================================

_WEATHER_CODE_MAP: dict[int, str] = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}

# Weather codes that increase subjective fatigue / homeostatic sleep drive
_SLEEP_PRESSURE_ELEVATING_CODES = {45, 48, 51, 53, 55, 61, 63, 65, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99}
_OVERCAST_CODES = {3, 45, 48}


# =============================================================================
# HEMISPHERE CLASSIFICATION
# =============================================================================

def _classify_hemisphere(latitude: float) -> str:
    """
    Classify hemisphere from latitude.

    Args:
        latitude: Decimal degrees. Positive = North, Negative = South.

    Returns:
        "northern", "southern", or "equatorial" (within ±10° of equator).
    """
    if abs(latitude) <= 10.0:
        return "equatorial"
    return "northern" if latitude > 0 else "southern"


# =============================================================================
# GEOLOCATION FROM IP (ip-api.com — free, no key, 45 req/min)
# =============================================================================

def fetch_location_from_ip(ip_address: str) -> dict:
    """
    Resolve geolocation from a client IP address using ip-api.com.

    Returns a dict with lat, lon, timezone, city, country — or defaults
    (Hyderabad, India) if the lookup fails or IP is private/loopback.

    Args:
        ip_address: IPv4 or IPv6 address string.

    Returns:
        {
            "lat": float,
            "lon": float,
            "timezone": str,   # e.g. "Asia/Kolkata"
            "city": str,
            "country": str,
        }
    """
    _default = {
        "lat": 17.38,
        "lon": 78.49,
        "timezone": "Asia/Kolkata",
        "city": "Hyderabad",
        "country": "India",
    }

    # Skip lookup for loopback / private IPs
    if not ip_address or ip_address in ("127.0.0.1", "::1", "localhost"):
        logger.debug("Private/loopback IP — using default location (Hyderabad)")
        return _default

    try:
        url = (
            f"http://ip-api.com/json/{ip_address}"
            f"?fields=status,lat,lon,timezone,city,country"
        )
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                logger.info(
                    f"📍 Geolocation resolved: {data.get('city')}, "
                    f"{data.get('country')} ({data.get('lat')}, {data.get('lon')})"
                )
                return {
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                    "timezone": data.get("timezone", "UTC"),
                    "city": data.get("city", "Unknown"),
                    "country": data.get("country", "Unknown"),
                }
    except Exception as exc:
        logger.warning(f"IP geolocation failed ({ip_address}): {exc}")

    return _default


# =============================================================================
# WEATHER CONTEXT FROM OPEN-METEO (free, no API key required)
# =============================================================================

def fetch_weather_context(lat: float, lon: float) -> dict:
    """
    Fetch current weather from Open-Meteo API.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        {
            "weather_code": int,
            "temperature_celsius": float,
            "weather_description": str,
            "elevates_sleep_pressure": bool,
        }
    """
    _default = {
        "weather_code": 0,
        "temperature_celsius": None,
        "weather_description": None,
        "elevates_sleep_pressure": False,
    }

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true&timezone=auto&forecast_days=1"
        )
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            cw = data.get("current_weather", {})
            code = int(cw.get("weathercode", 0))
            temp = cw.get("temperature")
            description = _WEATHER_CODE_MAP.get(code, "Unknown")
            elevates = code in _SLEEP_PRESSURE_ELEVATING_CODES

            logger.info(
                f"🌤 Weather: {description} | {temp}°C | "
                f"Sleep-pressure elevating: {elevates}"
            )
            return {
                "weather_code": code,
                "temperature_celsius": float(temp) if temp is not None else None,
                "weather_description": description,
                "elevates_sleep_pressure": elevates,
            }
    except Exception as exc:
        logger.warning(f"Weather fetch failed ({lat}, {lon}): {exc}")

    return _default


# =============================================================================
# CHRONOBIOLOGY ENGINE
# =============================================================================

class ChronobiologyEngine:
    """
    Deterministic inference engine for passive biological state estimation.

    Now integrates real geolocation (IP-based) and live weather data to
    produce a richer BiologicalState — including weather-influenced sleep
    pressure, location-aware hemisphere classification, and timezone-corrected
    circadian phase determination.

    All safety-critical logic remains rule-based — no ML models, no LLM calls.

    Usage:
        >>> engine = ChronobiologyEngine()
        >>> state = engine.infer_state_from_request("203.0.113.5")
        >>> state.location_city
        'Mumbai'
        >>> state.weather_description
        'Partly cloudy'
    """

    # Default location: Hyderabad, India (17.38°N, 78.49°E)
    DEFAULT_LATITUDE = 17.38
    DEFAULT_LONGITUDE = 78.49

    def __init__(self) -> None:
        logger.info("ChronobiologyEngine v2.0 initialized (geolocation + weather enabled)")

    # -------------------------------------------------------------------------
    # PUBLIC API — PRIMARY ENTRY POINT (with IP geolocation + weather)
    # -------------------------------------------------------------------------

    def infer_state_from_request(self, ip_address: str) -> BiologicalState:
        """
        Infer biological state from a client IP address.

        Performs geolocation lookup, fetches live weather, then computes
        all chronobiological dimensions. This is the preferred entry point
        for production use — it uses the user's real location and weather.

        Args:
            ip_address: The client's IP address from the HTTP request.

        Returns:
            BiologicalState with full geolocation + weather context.
        """
        geo = fetch_location_from_ip(ip_address)

        lat = geo["lat"]
        lon = geo["lon"]
        tz_name = geo.get("timezone", "UTC")

        # Resolve local time from timezone string
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(tz_name)
            local_time = datetime.now(tz)
        except Exception:
            # Fallback: UTC
            local_time = datetime.now(timezone.utc)

        weather = fetch_weather_context(lat, lon)

        return self.infer_state(
            local_time=local_time,
            coordinates=(lat, lon),
            weather_data=weather,
            location_city=geo.get("city"),
            location_country=geo.get("country"),
        )

    def infer_state(
        self,
        local_time: datetime,
        coordinates: Optional[Tuple[float, float]] = None,
        weather_data: Optional[dict] = None,
        location_city: Optional[str] = None,
        location_country: Optional[str] = None,
    ) -> BiologicalState:
        """
        Infer the user's biological state from local time and optional context.

        Args:
            local_time: The user's local datetime.
            coordinates: Optional (latitude, longitude) tuple.
                         Defaults to Hyderabad, India (17.38°N).
            weather_data: Optional dict from fetch_weather_context().
            location_city: City name for display purposes.
            location_country: Country name for display purposes.

        Returns:
            BiologicalState: Pydantic model with all inferred fields.
        """
        lat, lon = coordinates or (self.DEFAULT_LATITUDE, self.DEFAULT_LONGITUDE)
        hemisphere = _classify_hemisphere(lat)
        hour = local_time.hour
        month = local_time.month

        # Compute circadian dimensions
        phase = self._determine_circadian_phase(hour)
        season = self._determine_seasonal_influence(month, hemisphere)
        misaligned = self._detect_misalignment(hour)

        # Weather-influenced sleep pressure
        pressure = self._determine_sleep_pressure(
            hour,
            weather_elevates=weather_data.get("elevates_sleep_pressure", False) if weather_data else False,
        )

        # Compose advisory string (includes weather context)
        advisory = self._compose_advisory(
            phase, misaligned, season, hour, weather_data
        )

        state = BiologicalState(
            local_time=local_time,
            circadian_phase=phase,
            seasonal_influence=season,
            sleep_pressure_estimate=pressure,
            is_misaligned=misaligned,
            hemisphere=hemisphere,
            advisory=advisory,
            temperature_celsius=weather_data.get("temperature_celsius") if weather_data else None,
            weather_description=weather_data.get("weather_description") if weather_data else None,
            location_city=location_city,
            location_country=location_country,
        )

        logger.info(
            f"🕐 Inferred state @ {hour:02d}:00 | "
            f"Phase: {phase.value} | Season: {season.value} | "
            f"Pressure: {pressure.value} | Misaligned: {misaligned} | "
            f"Weather: {weather_data.get('weather_description') if weather_data else 'N/A'}"
        )

        return state

    # -------------------------------------------------------------------------
    # PRIVATE: CIRCADIAN PHASE DETERMINATION
    # -------------------------------------------------------------------------

    @staticmethod
    def _determine_circadian_phase(hour: int) -> CircadianPhase:
        """
        Map the local hour to a circadian phase.

        Phase Map:
            04–06: EARLY_MORNING    — Cortisol surge, parasympathetic→sympathetic
            07–09: MORNING_ACTIVATION — Peak cortisol, highest alertness
            10–11: LATE_MORNING     — Sustained focus, cognitive peak
            12–13: AFTERNOON_PEAK   — Digestive fire (agni) at maximum
            14–16: AFTERNOON_SLUMP  — Post-prandial dip, adenosine buildup
            17–18: EVENING_ACTIVE   — Second cortisol mini-peak
            19–21: WIND_DOWN        — Dim light melatonin onset (DLMO)
            22–03: DEEP_SLEEP       — Core body temp minimum, SWS dominant
        """
        if 4 <= hour <= 6:
            return CircadianPhase.EARLY_MORNING
        elif 7 <= hour <= 9:
            return CircadianPhase.MORNING_ACTIVATION
        elif 10 <= hour <= 11:
            return CircadianPhase.LATE_MORNING
        elif 12 <= hour <= 13:
            return CircadianPhase.AFTERNOON_PEAK
        elif 14 <= hour <= 16:
            return CircadianPhase.AFTERNOON_SLUMP
        elif 17 <= hour <= 18:
            return CircadianPhase.EVENING_ACTIVE
        elif 19 <= hour <= 21:
            return CircadianPhase.WIND_DOWN
        else:
            # 22–23 and 0–3
            return CircadianPhase.DEEP_SLEEP

    # -------------------------------------------------------------------------
    # PRIVATE: SEASONAL INFLUENCE DETERMINATION
    # -------------------------------------------------------------------------

    @staticmethod
    def _determine_seasonal_influence(
        month: int, hemisphere: str
    ) -> SeasonalInfluence:
        """
        Determine seasonal physiological influence from month and hemisphere.

        Uses Ayurvedic Ritucharya mapping with hemisphere inversion for
        the Southern Hemisphere.
        """
        if hemisphere == "equatorial":
            if 6 <= month <= 10:
                return SeasonalInfluence.MONSOON_DAMPNESS
            elif month in (11, 12, 1, 2):
                return SeasonalInfluence.WINTER_ACCUMULATION
            else:
                return SeasonalInfluence.SUMMER_HEAT

        northern_map = {
            12: SeasonalInfluence.WINTER_ACCUMULATION,
            1:  SeasonalInfluence.WINTER_ACCUMULATION,
            2:  SeasonalInfluence.WINTER_ACCUMULATION,
            3:  SeasonalInfluence.SPRING_RELEASE,
            4:  SeasonalInfluence.SPRING_RELEASE,
            5:  SeasonalInfluence.SUMMER_HEAT,
            6:  SeasonalInfluence.SUMMER_HEAT,
            7:  SeasonalInfluence.MONSOON_DAMPNESS,
            8:  SeasonalInfluence.MONSOON_DAMPNESS,
            9:  SeasonalInfluence.AUTUMN_TRANSITION,
            10: SeasonalInfluence.AUTUMN_TRANSITION,
            11: SeasonalInfluence.LATE_AUTUMN_DRY,
        }

        if hemisphere == "northern":
            return northern_map[month]

        # Southern Hemisphere: shift 6 months
        shifted_month = ((month - 1 + 6) % 12) + 1
        return northern_map[shifted_month]

    # -------------------------------------------------------------------------
    # PRIVATE: SLEEP PRESSURE ESTIMATION (weather-aware)
    # -------------------------------------------------------------------------

    @staticmethod
    def _determine_sleep_pressure(
        hour: int,
        weather_elevates: bool = False,
    ) -> SleepPressure:
        """
        Estimate homeostatic sleep drive from time of day and weather.

        Based on Borbély's Process S model — adenosine accumulates during
        wakefulness. Additionally, overcast/rainy weather is documented to
        increase subjective fatigue (dim-light melatonin effects, reduced
        alertness) and is reflected here as a pressure elevation.

        Weather elevation rule:
            LOW + rainy      → MODERATE
            MODERATE + rainy → HIGH
            HIGH             → HIGH (unchanged)

        Args:
            hour: Hour of day (0–23).
            weather_elevates: True if current weather promotes fatigue.

        Returns:
            SleepPressure enum value.
        """
        if 4 <= hour <= 11:
            base = SleepPressure.LOW
        elif 12 <= hour <= 18:
            base = SleepPressure.MODERATE
        else:
            base = SleepPressure.HIGH

        if not weather_elevates or base == SleepPressure.HIGH:
            return base

        # Upgrade one level when weather elevates fatigue
        if base == SleepPressure.LOW:
            return SleepPressure.MODERATE
        return SleepPressure.HIGH

    # -------------------------------------------------------------------------
    # PRIVATE: MISALIGNMENT DETECTION
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_misalignment(hour: int) -> bool:
        """
        Detect circadian misalignment.

        A user querying during biological sleep hours (22:00–04:00)
        is potentially experiencing insomnia, shift work, or jet lag.
        """
        return hour >= 22 or hour <= 4

    # -------------------------------------------------------------------------
    # PRIVATE: ADVISORY COMPOSITION (weather-aware)
    # -------------------------------------------------------------------------

    @staticmethod
    def _compose_advisory(
        phase: CircadianPhase,
        misaligned: bool,
        season: SeasonalInfluence,
        hour: int,
        weather_data: Optional[dict] = None,
    ) -> str:
        """
        Compose a contextual advisory string for downstream systems.

        Now includes weather context when available.
        """
        parts: list[str] = []

        # Misalignment advisory (highest priority)
        if misaligned:
            parts.append(
                f"⚠️ User is active at {hour:02d}:00 during the "
                f"'{phase.value}' phase. This suggests possible insomnia, "
                f"shift work, or acute distress. Prioritize calming, "
                f"sleep-supportive recommendations."
            )

        # Phase-specific advisories
        phase_notes = {
            CircadianPhase.EARLY_MORNING: (
                "Early morning — cortisol is surging. Warm beverages and "
                "gentle movement are well-tolerated. Avoid heavy remedies."
            ),
            CircadianPhase.MORNING_ACTIVATION: (
                "Peak cortisol window — ideal time for stimulating herbs "
                "(e.g., ginger, tulsi). Cognitive function is at its best."
            ),
            CircadianPhase.LATE_MORNING: (
                "Sustained focus period — cognitive performance peaks. "
                "Adaptogenic herbs are most effective during this window."
            ),
            CircadianPhase.AFTERNOON_PEAK: (
                "Digestive fire (agni) is at maximum — optimal absorption "
                "for oral remedies and supplements."
            ),
            CircadianPhase.AFTERNOON_SLUMP: (
                "Post-prandial dip — adenosine building, alertness drops. "
                "Avoid sedating remedies; consider peppermint or light spices."
            ),
            CircadianPhase.EVENING_ACTIVE: (
                "Second wind period — gentle exercise well-tolerated. "
                "Begin transitioning to calming remedies."
            ),
            CircadianPhase.WIND_DOWN: (
                "Melatonin onset phase — favor calming herbs (chamomile, "
                "valerian, lavender). Avoid stimulants."
            ),
            CircadianPhase.DEEP_SLEEP: (
                "Biological repair phase — sleep-supportive remedies only. "
                "Avoid anything stimulating."
            ),
        }
        if not misaligned:
            parts.append(phase_notes.get(phase, ""))

        # Seasonal advisories
        season_notes = {
            SeasonalInfluence.WINTER_ACCUMULATION: (
                "Winter (Kapha season): warming spices favored. "
                "Immunity is naturally strong; heavier foods tolerated."
            ),
            SeasonalInfluence.SPRING_RELEASE: (
                "Spring (Kapha release): detox-supportive herbs favored. "
                "Watch for allergies and respiratory congestion."
            ),
            SeasonalInfluence.SUMMER_HEAT: (
                "Summer (Pitta season): cooling remedies preferred "
                "(mint, fennel, aloe). Hydration is critical."
            ),
            SeasonalInfluence.MONSOON_DAMPNESS: (
                "Monsoon (Vata aggravation): digestion weakened. "
                "Warm, light foods and digestive spices are ideal."
            ),
            SeasonalInfluence.AUTUMN_TRANSITION: (
                "Autumn (residual Pitta): anti-inflammatory herbs helpful. "
                "Skin and digestive issues may flare."
            ),
            SeasonalInfluence.LATE_AUTUMN_DRY: (
                "Late autumn (Vata dominant): moisturizing, grounding "
                "remedies favored. Joint pain and anxiety may increase."
            ),
        }
        parts.append(season_notes.get(season, ""))

        # Weather advisory
        if weather_data:
            desc = weather_data.get("weather_description")
            temp = weather_data.get("temperature_celsius")
            elevates = weather_data.get("elevates_sleep_pressure", False)

            if desc:
                weather_note = f"Current weather: {desc}"
                if temp is not None:
                    weather_note += f" ({temp:.1f}°C)"
                if elevates:
                    weather_note += (
                        " — gloomy conditions may increase fatigue; "
                        "favour warming or energising remedies."
                    )
                parts.append(weather_note)

        return " | ".join(p for p in parts if p)
