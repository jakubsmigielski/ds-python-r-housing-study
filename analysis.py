import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib

matplotlib.use('TkAgg')


FILE_PATH = 'data/Houses.csv'
TARGET_COLUMNS = ['price', 'sq', 'rooms', 'city']
REPORT_DATA = {}



def load_and_prepare_data():
    print("--- 1. Data Loading and Preparation ---")
    try:
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"File not found at: {os.path.abspath(FILE_PATH)}")

        df = pd.read_csv(FILE_PATH, encoding='windows-1250')
        df_clean = df[TARGET_COLUMNS].dropna(subset=['price', 'sq', 'rooms']).copy()

        df_final = df_clean[
            (df_clean['sq'] > 10) & (df_clean['sq'] < 400) &
            (df_clean['rooms'] >= 1) &
            (df_clean['price'] > 1000) &
            (df_clean['price'] < 5000000)
            ].copy()

        df_final['rooms'] = df_final['rooms'].astype(int)
        df_final['city'] = df_final['city'].astype('category')
        df_final['price_log'] = np.log(df_final['price'])

        REPORT_DATA['Initial Rows'] = len(df)
        REPORT_DATA['Final Rows'] = len(df_final)

        print(f" Final dataset size: {len(df_final)} rows.")
        return df_final
    except Exception as e:
        print(f" Error during loading,cleaning: {e}")
        return None


def visualize_data(df):
    print("\n--- 2. Data Visualization ---")
    sns.set_style("whitegrid");
    sns.set_palette("Set2")

    # plotttt1
    plt.figure(figsize=(10, 6))
    sns.regplot(x='sq', y='price_log', data=df, scatter_kws={'alpha': 0.6, 's': 20}, line_kws={'color': 'red'})
    plt.title('Relationship Between Log(Price) and Area')
    plt.xlabel('Area [sq m]');
    plt.ylabel('Log(Price)')
    plt.show()

    # plotttt2
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='city', y='price', data=df, inner='quartile', cut=0, scale='width')
    plt.title('Price Distribution Across Cities')
    plt.xlabel('City');
    plt.ylabel('Price [PLN]')
    plt.show()


def train_and_analyze_model(df):
    print("\n--- 3. Model Training (Random Forest) ---")

    X = df[['sq', 'rooms', 'city']].copy()
    y = df['price_log'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['city']),
            ('scaler', StandardScaler(), ['sq', 'rooms'])
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    model_pipeline.fit(X_train, y_train)


    y_pred_log = model_pipeline.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_test_original = np.exp(y_test)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    r2 = r2_score(y_test_original, y_pred)

    REPORT_DATA['RMSE'] = rmse
    REPORT_DATA['R2'] = r2

    print(f"Random Forest Model: R2={r2:.4f}, RMSE={rmse:.2f} PLN")

    # Feature Importance
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = model_pipeline.named_steps['regressor'].feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace('^.*__', '', regex=True)

    city_importance = feature_importance_df[feature_importance_df['Feature'].str.contains('city')]['Importance'].sum()

    final_importances = feature_importance_df[~feature_importance_df['Feature'].str.contains('city')]
    final_importances = pd.concat(
        [final_importances, pd.DataFrame([{'Feature': 'city', 'Importance': city_importance}])], ignore_index=True)
    final_importances = final_importances.sort_values(by='Importance', ascending=False)

    REPORT_DATA['Feature Importance'] = final_importances


    residuals = y_test_original - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='red', linestyles='--')
    plt.title(f'Residual Plot for Random Forest Model')
    plt.xlabel('Predicted Price [PLN]');
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.show()

    return model_pipeline


def display_combined_report():
    print("\n--- 4. Visual Report ---")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2.5]})

    # A. MODEL METRICS (Table)
    data = [[f"{REPORT_DATA.get('R2', 0):.4f}"], [f"{REPORT_DATA.get('RMSE', 0) / 1000:.0f}K PLN"]]
    ax0 = axes[0];
    ax0.set_title(f"Model Performance Summary", fontsize=12, pad=15)
    ax0.axis('tight');
    ax0.axis('off')
    header_color = '#333333'
    table = ax0.table(cellText=data, colLabels=['RandomForest'], rowLabels=['R-squared (R²)', 'RMSE'],
                      cellLoc='center', loc='center', cellColours=[['#d4edda']] * 2, colColours=[header_color] * 1
                      )
    table.auto_set_font_size(False);
    table.set_fontsize(10);
    table.scale(1.0, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(color='white')

    ax1 = axes[1]
    importance_df = REPORT_DATA['Feature Importance']
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax1, palette='viridis')
    ax1.set_title('Feature Importance (What drives the price)', fontsize=12)
    ax1.set_xlabel('Relative Importance Score');
    ax1.set_ylabel('')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    final_df = load_and_prepare_data()

    if final_df is not None and not final_df.empty:

        visualize_data(final_df)

        final_pipeline = train_and_analyze_model(final_df)

        display_combined_report()

        if final_pipeline:
            print("\n--- FINAL PREDICTION ---")


            X = final_df[['sq', 'rooms', 'city']].copy()
            y_log = final_df['price_log'].copy()
            final_pipeline.fit(X, y_log)

            input_data = {'Area [sq m]': [65], 'Rooms': [3], 'City': ['Kraków']}

            new_data = pd.DataFrame({'sq': [65], 'rooms': [3], 'city': ['Kraków']})
            new_data['city'] = new_data['city'].astype('category')

            predicted_price_log = final_pipeline.predict(new_data)
            predicted_price = np.exp(predicted_price_log[0])

            results = pd.DataFrame({
                'Feature': ['Area [sq m]', 'Rooms', 'City', 'Predicted Price (PLN)'],
                'Value': [
                    input_data['Area [sq m]'][0], input_data['Rooms'][0], input_data['City'][0],
                    f"{predicted_price:,.0f}"
                ]
            })

            print("Input Data and Prediction Result:")
            print(results.to_markdown(index=False, tablefmt="pipe"))
            print("=" * 50)