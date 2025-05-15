import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_kmeans_decision_tree(df, k=3):
    """
    Thực hiện phân tích dữ liệu kết hợp KMeans và Decision Tree:
    - Tiền xử lý NaN
    - Chọn 2 cột số để trực quan
    - Chuẩn hóa dữ liệu
    - Phân cụm KMeans
    - Mã hóa đặc trưng để huấn luyện Decision Tree
    - Trực quan phân cụm và cây quyết định
    - Lưu hình ảnh kết quả
    Trả về đường dẫn hình ảnh đã lưu.
    """

    # --------------------------------------------
    # Bước 1: Tiền xử lý NaN
    # --------------------------------------------
    df = df.copy()
    # Điền NaN cột số bằng trung bình
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    # Điền NaN cột object bằng mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # --------------------------------------------
    # Bước 2: Kiểm tra tối thiểu dữ liệu
    # --------------------------------------------
    min_samples_required = max(10, k * 3)
    if len(df) < min_samples_required:
        raise ValueError(f"Cần ít nhất {min_samples_required} mẫu dữ liệu để phân tích.")
    
    # Tạo thư mục static để lưu hình
    os.makedirs("static", exist_ok=True)

    # --------------------------------------------
    # Bước 3: Chọn 2 cột số để trực quan
    # --------------------------------------------
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    if df_numeric.shape[1] < 2:
        raise ValueError("⚠️ Cần ít nhất 2 cột số để thực hiện phân tích trực quan.")
    df_numeric = df_numeric.iloc[:, :2]  # lấy 2 cột đầu

    # --------------------------------------------
    # Bước 4: Chuẩn hóa dữ liệu
    # --------------------------------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    # --------------------------------------------
    # Bước 5: Điều chỉnh số cụm phù hợp
    # --------------------------------------------
    k = min(k, len(df) // 3)  # đảm bảo không quá lớn so với số mẫu

    # --------------------------------------------
    # Bước 6: Phân cụm KMeans
    # --------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = clusters  # gán nhãn cụm vào DataFrame gốc

    # --------------------------------------------
    # Bước 7: Mã hóa đặc trưng cho Decision Tree
    # --------------------------------------------
    df_encoded = df.copy()
    le_dict = {}  # lưu các LabelEncoder nếu có
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le

    # Chuẩn bị dữ liệu huấn luyện
    X = df_encoded.drop(columns=['Cluster'])
    y = df_encoded['Cluster']

    # --------------------------------------------
    # Bước 8: Chia dữ liệu train/test phù hợp
    # --------------------------------------------
    test_size = max(0.2, k * 2 / len(df))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    except ValueError:
        # Trường hợp stratify không phù hợp (dữ liệu nhỏ hoặc phân lớp không đủ)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # --------------------------------------------
    # Bước 9: Huấn luyện cây quyết định
    # --------------------------------------------
    max_depth = min(3, int(np.log2(len(X_train))) if len(X_train) > 1 else 1)
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    )
    clf.fit(X_train, y_train)

    # --------------------------------------------
    # Bước 10: Trực quan kết quả
    # --------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    markers = ['o', 's', '^', 'D', 'v']

    # Vẽ phân cụm KMeans
    for i in range(k):
        mask = clusters == i
        ax1.scatter(
            scaled_data[mask, 0], scaled_data[mask, 1],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f'Cụm {i+1}',
            s=100,
            alpha=0.7,
            edgecolors='white'
        )

    # Tâm cụm
    centers = kmeans.cluster_centers_
    ax1.scatter(
        centers[:, 0], centers[:, 1],
        c='yellow', marker='*', s=300,
        label='Tâm cụm', edgecolors='black'
    )

    ax1.set_title("Phân cụm K-Means", fontsize=14)
    ax1.set_xlabel(df_numeric.columns[0])
    ax1.set_ylabel(df_numeric.columns[1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Vẽ cây quyết định
    plot_tree(clf, feature_names=X.columns, class_names=[f'Cụm {i+1}' for i in range(k)],
              filled=True, rounded=True, ax=ax2, fontsize=10)
    ax2.set_title("Cây quyết định phân loại cụm", fontsize=14)

    # --------------------------------------------
    # Bước 11: Thống kê và chú thích kết quả
    # --------------------------------------------
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    analysis_text = f"""
    Thông tin phân tích:
    - Số mẫu: {len(df)}
    - Số cụm: {k}
    - Độ chính xác (train): {train_score:.2f}
    - Độ chính xác (test): {test_score:.2f}
    Phân bố cụm:"""
    for i in range(k):
        count = np.sum(clusters == i)
        percent = count / len(df) * 100
        analysis_text += f"\nCụm {i+1}: {count} mẫu ({percent:.1f}%)"
    plt.figtext(0.02, 0.02, analysis_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # --------------------------------------------
    # Bước 12: Lưu hình và trả về đường dẫn
    # --------------------------------------------
    img_path = f'static/combined_{uuid.uuid4().hex}.png'
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    return img_path