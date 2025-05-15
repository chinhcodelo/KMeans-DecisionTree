import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans(df, k=3):
    """
    Phân cụm KMeans, xử lý NaN bằng cách điền trung bình.
    """
    df = df.copy()

    # Điền NaN bằng trung bình
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Chọn 2 cột để trực quan
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    if df_numeric.shape[1] < 2:
        raise ValueError("⚠️ Cần ít nhất 2 cột số để làm dữ liệu phân cụm.")
    df_numeric = df_numeric.iloc[:, :2]

    # Chuẩn hóa
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    # Điều chỉnh k
    k = min(k, len(df) // 2)

    # Phân cụm
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(scaled_data)

    # Thêm nhãn
    df['Cluster'] = clusters

    # Vẽ
    plt.figure(figsize=(12,8))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    markers = ['o', 's', '^', 'D', 'v']
    for cluster_id in range(k):
        mask = clusters == cluster_id
        plt.scatter(
            scaled_data[mask, 0], scaled_data[mask, 1],
            c=colors[cluster_id % len(colors)],
            marker=markers[cluster_id % len(markers)],
            label=f'Cụm {cluster_id+1}', s=100, alpha=0.7, edgecolors='white'
        )
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1], c='yellow', marker='*', s=300, label='Tâm cụm', edgecolors='black')

    plt.title(f"Kết quả phân cụm K-Means (k={k})", fontsize=16)
    plt.xlabel(df_numeric.columns[0])
    plt.ylabel(df_numeric.columns[1])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Thông tin
    stats_text = "Thông tin phân cụm:\n"
    stats_text += f"Tổng số mẫu: {len(df)}\n"
    for i in range(k):
        count = np.sum(clusters == i)
        percent = count / len(df) * 100
        stats_text += f"Cụm {i+1}: {count} mẫu ({percent:.1f}%)\n"
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    # Lưu
    os.makedirs("static", exist_ok=True)
    img_path = f'static/kmeans_{uuid.uuid4().hex}.png'
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    return img_path