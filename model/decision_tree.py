import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run_decision_tree(df):
    """
    Xây dựng và trực quan hóa cây quyết định:
    - Tiền xử lý NaN
    - Chuyển đổi categorical thành số
    - Phân chia dữ liệu train/test
    - Huấn luyện cây quyết định
    - Vẽ cây, hiển thị thông tin
    - Trả về đường dẫn hình ảnh
    """

    df = df.copy()

    # Tiền xử lý NaN
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Kiểm tra đủ mẫu
    if len(df) < 10:
        raise ValueError("Cần ít nhất 10 mẫu dữ liệu để xây dựng cây quyết định.")

    # Chuẩn bị dữ liệu
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le_dict = {}
    # Mã hóa y nếu là object
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        le_dict['target'] = le_y

    # Mã hóa X nếu có categorical
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le

    # Tính test_size dựa trên số lớp
    n_classes = len(np.unique(y))
    min_samples_per_class = 2
    min_test_size = (n_classes * min_samples_per_class) / len(y)
    test_size = max(0.2, min_test_size)

    # Chia dữ liệu train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Đặt max_depth hợp lý
    max_depth = min(4, int(np.log2(len(X_train))) if len(X_train) > 1 else 1)

    # Huấn luyện cây
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    )
    clf.fit(X_train, y_train)

    # Đánh giá
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    # Trực quan hóa
    plt.figure(figsize=(20,12))
    plot_tree(clf, feature_names=X.columns, class_names=[str(c) for c in sorted(np.unique(y))],
              filled=True, rounded=True, fontsize=14, precision=2)
    plt.title("Cây quyết định", fontsize=20, pad=20)

    # Thông tin mô hình
    info_text = f"""
    Thông tin mô hình:
    - Độ sâu tối đa: {clf.get_depth()}
    - Số nút lá: {clf.get_n_leaves()}
    - Số mẫu huấn luyện: {len(X_train)}
    - Số mẫu kiểm tra: {len(X_test)}
    - Độ chính xác (train): {train_score:.2f}
    - Độ chính xác (test): {test_score:.2f}
    """
    plt.figtext(0.02, 0.02, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Lưu ảnh
    os.makedirs("static", exist_ok=True)
    img_path = f'static/tree_{uuid.uuid4().hex}.png'
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    return img_path