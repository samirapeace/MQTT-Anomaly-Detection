import numpy as np
import pandas as pd

class FeatureEngineer:

    def __init__(self):
        pass

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # =========================
        # 1. BASIC SAFETY
        # =========================
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # =========================
        # 2. MESSAGE BEHAVIOR
        # =========================
        if "mqtt.msgtype" in df.columns:
            df["msgtype_freq"] = df.groupby("mqtt.msgtype")["mqtt.msgtype"].transform("count")

        if "mqtt.qos" in df.columns:
            df["qos_freq"] = df.groupby("mqtt.qos")["mqtt.qos"].transform("count")

        if "mqtt.dupflag" in df.columns and "mqtt.retain" in df.columns:
            df["flag_sum"] = df["mqtt.dupflag"] + df["mqtt.retain"]

        # =========================
        # 3. TEMPORAL FEATURES
        # =========================
        if "tcp.time_delta" in df.columns:
            df["msg_rate"] = 1 / (df["tcp.time_delta"] + 1e-6)
            df["fast_packet"] = (df["tcp.time_delta"] < 0.01).astype(int)

        # =========================
        # 4. ROLLING FEATURES
        # =========================
        if "mqtt.qos" in df.columns:
            df["rolling_qos_mean"] = df["mqtt.qos"].rolling(10, min_periods=1).mean()

        if "mqtt.msgtype" in df.columns:
            df["rolling_msgtype_mean"] = df["mqtt.msgtype"].rolling(10, min_periods=1).mean()

        # =========================
        # 5. RATIO FEATURES
        # =========================
        if "mqtt.qos" in df.columns and "mqtt.len" in df.columns:
            df["qos_to_len"] = df["mqtt.qos"] / (df["mqtt.len"] + 1)

        if "mqtt.proto_len" in df.columns and "mqtt.len" in df.columns:
            df["proto_ratio"] = df["mqtt.proto_len"] / (df["mqtt.len"] + 1)

        # =========================
        # 6. SESSION FEATURES
        # =========================
        if "mqtt.kalive" in df.columns:
            mean_kalive = df["mqtt.kalive"].mean()
            df["keepalive_anomaly"] = (df["mqtt.kalive"] > mean_kalive).astype(int)

        if "mqtt.conflag.cleansess" in df.columns:
            df["clean_session_flag"] = df["mqtt.conflag.cleansess"]

        # =========================
        # 7. SUBSCRIPTION BEHAVIOR
        # =========================
        if "mqtt.sub.qos" in df.columns and "mqtt.suback.qos" in df.columns:
            df["sub_qos_diff"] = df["mqtt.sub.qos"] - df["mqtt.suback.qos"]

        # =========================
        # 8. WILL MESSAGE FEATURES
        # =========================
        if "mqtt.willmsg_len" in df.columns and "mqtt.willtopic_len" in df.columns:
            df["will_size_ratio"] = df["mqtt.willmsg_len"] / (df["mqtt.willtopic_len"] + 1)

        # =========================
        # 9. TCP BEHAVIOR
        # =========================
        if "tcp.flags" in df.columns:
            df["tcp_flag_bin"] = df["tcp.flags"].astype("category").cat.codes

        if "tcp.len" in df.columns:
            df["large_packet"] = (df["tcp.len"] > df["tcp.len"].mean()).astype(int)

        # =========================
        # 10. INTERACTION FEATURES
        # =========================
        if "mqtt.msgtype" in df.columns and "mqtt.qos" in df.columns:
            df["msg_qos_interaction"] = df["mqtt.msgtype"] * df["mqtt.qos"]

        if "mqtt.dupflag" in df.columns and "mqtt.msgtype" in df.columns:
            df["dup_msg_interaction"] = df["mqtt.dupflag"] * df["mqtt.msgtype"]

        # =========================
        # FINAL CLEAN
        # =========================
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        return df