"""
Makine Öğrenmesi ile Meme Tümörü Sınıflandırma

Amaç:
Hücre çekirdeğine ait ölçümleri kullanarak tümörün
iyi huylu (Benign) veya kötü huylu (Malignant) olduğunu
makine öğrenmesi ile tahmin etmek.

Kullanılan kütüphaneler:
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Proje adımları:
1. Veri setinin yüklenmesi ve incelenmesi
2. Eksik ve aykırı değer kontrolü
3. Öznitelik mühendisliği
4. Öznitelik seçimi
5. Train / Validation / Test ayrımı
6. Veri ölçekleme
7. Farklı modellerin karşılaştırılması
8. 5-Fold Cross Validation
9. Grid Search ile hiperparametre ayarlama
10. Test değerlendirmesi
11. Model katsayılarının incelenmesi

Çalıştırma:
Terminalde proje klasörüne girildikten sonra:

    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def main():

    # ================================================================
    # 1. VERİ SETİNİN YÜKLENMESİ VE İNCELENMESİ
    # ================================================================

    print("\n" + "=" * 70)
    print("1. VERİ SETİNİN YÜKLENMESİ VE İNCELENMESİ")
    print("=" * 70)

    # Scikit-learn içindeki Breast Cancer Wisconsin veri seti
    data = load_breast_cancer()

    df = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # Scikit-learn'de:
    # 0 = Malignant
    # 1 = Benign
    #
    # Projede daha anlaşılır olması için:
    # 0 = Benign
    # 1 = Malignant
    df["target"] = (data.target == 0).astype(int)

    print("Örnek sayısı:", len(df))
    print("Öznitelik sayısı:", len(data.feature_names))
    print("Hedef: 0 = Benign, 1 = Malignant")

    print("\nİlk 5 satır:")
    print(df.head())

    print("\nVeri setinin boyutu:")
    print(df.shape)

    print("\nVeri tipleri:")
    print(df.dtypes)

    print("\nTemel istatistikler:")
    print(df.describe())

    print("\nSınıf dağılımı:")
    print(df["target"].value_counts().sort_index())


    # ================================================================
    # 2. EKSİK VE AYKIRI DEĞER ANALİZİ
    # ================================================================

    print("\n" + "=" * 70)
    print("2. EKSİK VE AYKIRI DEĞER ANALİZİ")
    print("=" * 70)

    # Eksik değer kontrolü
    missing_values = df.isnull().sum().sum()

    print("Eksik değer sayısı:", int(missing_values))

    # IQR yöntemiyle potansiyel aykırı değerleri inceleme
    outlier_count = 0

    for column in data.feature_names:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        outliers = (
            (df[column] < lower_limit)
            |
            (df[column] > upper_limit)
        )

        outlier_count += int(outliers.sum())

    print(
        "IQR yöntemiyle potansiyel aykırı gözlem sayısı:",
        outlier_count
    )

    print(
        "Aykırı değerler silinmemiştir; "
        "bunların gerçek biyolojik ölçümlerdeki "
        "değişkenliği temsil edebileceği düşünülmüştür."
    )


    # ================================================================
    # 3. ÖZNİTELİK MÜHENDİSLİĞİ
    # ================================================================

    print("\n" + "=" * 70)
    print("3. ÖZNİTELİK MÜHENDİSLİĞİ")
    print("=" * 70)

    # Sıfıra bölme ihtimaline karşı küçük bir değer
    epsilon = 1e-6

    # Yeni özellik 1
    df["radius_texture_ratio"] = (
        df["mean radius"]
        /
        (df["mean texture"] + epsilon)
    )

    # Yeni özellik 2
    df["perimeter_area_ratio"] = (
        df["mean perimeter"]
        /
        (df["mean area"] + epsilon)
    )

    print("Oluşturulan yeni özellikler:")
    print("- radius_texture_ratio")
    print("- perimeter_area_ratio")


    # ================================================================
    # 4. HEDEF VE GİRDİ DEĞİŞKENLERİNİN BELİRLENMESİ
    # ================================================================

    X = df.drop(columns="target")
    y = df["target"]


    # ================================================================
    # 5. TRAIN / VALIDATION / TEST AYRIMI
    # ================================================================

    print("\n" + "=" * 70)
    print("4. VERİ AYRIMI")
    print("=" * 70)

    # %70 Train, %30 geçici veri
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Geçici verinin yarısı Validation,
    # yarısı Test olarak kullanılıyor.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]} (%70)")
    print(f"Validation: {X_val.shape[0]} (%15)")
    print(f"Test: {X_test.shape[0]} (%15)")


    # ================================================================
    # 6. MODELLER
    # ================================================================

    # SelectKBest Pipeline içinde kullanılıyor.
    #
    # Böylece feature selection sadece eğitim verisinden öğreniliyor.
    # Cross Validation sırasında da veri sızıntısı engelleniyor.

    models = {

        "Logistic Regression": Pipeline([
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=10
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=2000
                )
            )
        ]),

        "KNN": Pipeline([
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=10
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ]),

        "Decision Tree": Pipeline([
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=10
                )
            ),

            (
                "model",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                )
            )
        ]),

        "Random Forest": Pipeline([
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=10
                )
            ),

            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE
                )
            )
        ])
    }


    # ================================================================
    # 7. MODEL KARŞILAŞTIRMASI
    # ================================================================

    print("\n" + "=" * 70)
    print("5. MODEL KARŞILAŞTIRMASI")
    print("=" * 70)

    results = []

    for name, model in models.items():

        # Modeli eğitim verisiyle eğit
        model.fit(X_train, y_train)

        # Validation tahmini
        prediction = model.predict(X_val)

        # Sonuçları kaydet
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_val,
                prediction
            ),
            "Precision": precision_score(
                y_val,
                prediction,
                zero_division=0
            ),
            "Recall": recall_score(
                y_val,
                prediction,
                zero_division=0
            ),
            "F1-Score": f1_score(
                y_val,
                prediction,
                zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    # F1-Score'a göre büyükten küçüğe sırala
    results_df = results_df.sort_values(
        "F1-Score",
        ascending=False
    )

    print(
        results_df.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    # En yüksek validation F1 skoruna sahip model
    best_name = results_df.iloc[0]["Model"]

    print(
        "\nValidation sonucuna göre en başarılı model:",
        best_name
    )


    # ================================================================
    # 8. MODEL KARŞILAŞTIRMA GRAFİĞİ
    # ================================================================

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        results_df["Model"],
        results_df["F1-Score"]
    )

    plt.title(
        "Modellerin Validation F1-Skor Karşılaştırması"
    )

    plt.ylabel("F1-Score")

    plt.ylim(
        max(
            0,
            results_df["F1-Score"].min() - 0.08
        ),
        1
    )

    plt.xticks(rotation=15)

    for bar, value in zip(
        bars,
        results_df["F1-Score"]
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        "model_karsilastirma.png",
        dpi=300
    )

    plt.close()


    # ================================================================
    # 9. 5-FOLD CROSS VALIDATION
    # ================================================================

    print("\n" + "=" * 70)
    print("6. 5-FOLD CROSS VALIDATION")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_results = {}

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        cv_results[name] = scores

        print(
            f"{name:<20}: "
            f"F1 = {scores.mean():.4f} "
            f"(+/- {scores.std():.4f})"
        )


    # ================================================================
    # 10. GRID SEARCH
    # ================================================================

    print("\n" + "=" * 70)
    print(f"7. GRID SEARCH - {best_name}")
    print("=" * 70)

    # Her model için kullanılabilecek hiperparametreler
    param_grids = {

        "Logistic Regression": {
            "model__C": [
                0.1,
                1.0,
                10.0,
                50.0
            ]
        },

        "KNN": {
            "model__n_neighbors": [
                3,
                5,
                7,
                9
            ],

            "model__weights": [
                "uniform",
                "distance"
            ]
        },

        "Decision Tree": {
            "model__max_depth": [
                3,
                5,
                7,
                None
            ]
        },

        "Random Forest": {
            "model__n_estimators": [
                50,
                100,
                150
            ],

            "model__max_depth": [
                None,
                5,
                10
            ]
        }
    }

    grid_search = GridSearchCV(
        estimator=models[best_name],
        param_grid=param_grids[best_name],
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    # Grid Search sadece train verisi üzerinde çalışıyor
    grid_search.fit(
        X_train,
        y_train
    )

    final_model = grid_search.best_estimator_

    print(
        "En iyi parametreler:",
        grid_search.best_params_
    )

    print(
        f"En iyi CV F1: "
        f"{grid_search.best_score_:.4f}"
    )


    # ================================================================
    # 11. TEST SONUÇLARI
    # ================================================================

    print("\n" + "=" * 70)
    print("8. TEST SONUÇLARI")
    print("=" * 70)

    # Test verisi daha önce modele gösterilmedi.
    y_pred = final_model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    test_precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(
        f"Accuracy : "
        f"{test_accuracy:.4f} "
        f"(%{test_accuracy * 100:.2f})"
    )

    print(
        f"Precision: "
        f"{test_precision:.4f} "
        f"(%{test_precision * 100:.2f})"
    )

    print(
        f"Recall   : "
        f"{test_recall:.4f} "
        f"(%{test_recall * 100:.2f})"
    )

    print(
        f"F1-Score : "
        f"{test_f1:.4f} "
        f"(%{test_f1 * 100:.2f})"
    )


    # ================================================================
    # 12. CONFUSION MATRIX
    # ================================================================

    confusion = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(confusion)

    print(
        f"TN = {confusion[0, 0]}"
    )

    print(
        f"FP = {confusion[0, 1]}"
    )

    print(
        f"FN = {confusion[1, 0]}"
    )

    print(
        f"TP = {confusion[1, 1]}"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=[
            "Benign (0)",
            "Malignant (1)"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4.8)
    )

    display.plot(
        ax=ax,
        values_format="d",
        cmap="Blues",
        colorbar=True
    )

    ax.set_title(
        f"Test Hata Matrisi ({best_name})"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300
    )

    plt.close()


    # ================================================================
    # 13. ÖZNİTELİK ÖNEMİ
    # ================================================================

    print("\n" + "=" * 70)
    print("9. ÖZNİTELİK ÖNEMLERİ")
    print("=" * 70)

    # Pipeline içerisindeki feature selection aşamasını al
    selector = final_model.named_steps["selector"]

    selected_features = (
        X.columns[
            selector.get_support()
        ].tolist()
    )

    # Optimize edilmiş model
    fitted_model = final_model.named_steps["model"]

    # Logistic Regression için katsayıların mutlak değeri
    if hasattr(fitted_model, "coef_"):

        importance = np.abs(
            fitted_model.coef_[0]
        )

        importance_label = (
            "Mutlak Logistic Regression Katsayısı"
        )

    # Ağaç tabanlı modeller için feature importance
    elif hasattr(
        fitted_model,
        "feature_importances_"
    ):

        importance = (
            fitted_model.feature_importances_
        )

        importance_label = (
            "Öznitelik Önem Skoru"
        )

    else:

        importance = np.zeros(
            len(selected_features)
        )

        importance_label = (
            "Önem Skoru"
        )

    feature_df = pd.DataFrame({

        "Öznitelik": selected_features,

        "Önem": importance

    }).sort_values(
        "Önem",
        ascending=False
    )

    print(
        feature_df.to_string(
            index=False,
            float_format="%.4f"
        )
    )


    # ================================================================
    # 14. ÖZNİTELİK ÖNEM GRAFİĞİ
    # ================================================================

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.barh(
        feature_df["Öznitelik"][::-1],
        feature_df["Önem"][::-1]
    )

    plt.title(
        f"Özniteliklerin Etkisi - {best_name}"
    )

    plt.xlabel(
        importance_label
    )

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png",
        dpi=300
    )

    plt.close()


    # ================================================================
    # 15. PROJE SONUCU
    # ================================================================

    print("\n" + "=" * 70)
    print("10. PROJE SONUCU")
    print("=" * 70)

    print(
        f"En başarılı model: {best_name}"
    )

    print(
        f"Test Accuracy: "
        f"%{test_accuracy * 100:.2f}"
    )

    print(
        f"Test Precision: "
        f"%{test_precision * 100:.2f}"
    )

    print(
        f"Test Recall (Malignant): "
        f"%{test_recall * 100:.2f}"
    )

    print(
        f"Test F1-Score: "
        f"%{test_f1 * 100:.2f}"
    )

    print(
        "\nModel yalnızca eğitim amacıyla "
        "hazırlanmıştır. Klinik teşhis amacıyla "
        "kullanılamaz ve doktor kararının "
        "yerine geçmez."
    )

    print(
        "\nProje tamamlandı."
    )

    print(
        "Grafikler proje klasörüne kaydedildi:"
    )

    print("- model_karsilastirma.png")
    print("- confusion_matrix.png")
    print("- feature_importance.png")


# Programı çalıştır
if __name__ == "__main__":
    main()