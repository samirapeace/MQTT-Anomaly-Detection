import pandas as pd
import numpy as np

WINDOW = 1.0  # نافذة زمنية = 1 ثانية

def build_features(csv_file, label):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["frame.time_epoch"])
    df["time"] = df["frame.time_epoch"].astype(float)
    df["window"] = (df["time"] // WINDOW)

    rows = []

    for w, g in df.groupby("window"):
        pkt_count = len(g)
        rows.append({
            "packet_count": pkt_count,
            "packet_rate": pkt_count / WINDOW,
            "avg_tcp_len": g["tcp.len"].mean(),
            "std_tcp_len": g["tcp.len"].std(),
            "mqtt_publish_count": (g["mqtt.msgtype"] == 3).sum(),
            "unique_topics": g["mqtt.topic"].nunique(),
            "qos0_ratio": (g["mqtt.qos"] == 0).sum() / pkt_count if pkt_count > 0 else 0,
            "label": label
        })

    return pd.DataFrame(rows)

normal_df = build_features("normal_raw.csv", label=0)
attack_df = build_features("attack_raw.csv", label=1)

dataset = pd.concat([normal_df, attack_df], ignore_index=True)
dataset.to_csv("mqtt_dataset.csv", index=False)

print("Dataset created:", dataset.shape)
