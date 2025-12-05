import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import os


# 1. CẤU HÌNH & XỬ LÝ DỮ LIỆU

FILE_PATH_HISTORY = 'E:/allPythonProject/AIOProject/M06/data/FPT_train.csv'       
FILE_PATH_PREDICTION = 'E:/allPythonProject/AIOProject/M06/outputs/submission_120d_v4.csv' 

def load_and_process_data():
    # --- BƯỚC 1: Đọc dữ liệu Lịch sử (Input) ---
    df_hist = pd.read_csv(FILE_PATH_HISTORY)
    df_hist.columns = [c.lower().strip() for c in df_hist.columns]
    if 'time' in df_hist.columns: df_hist = df_hist.rename(columns={'time': 'Date'})
    
    rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
    df_hist = df_hist.rename(columns=rename_map)
    
    df_hist['Date'] = pd.to_datetime(df_hist['Date'])
    df_hist = df_hist.sort_values('Date')

    # --- BƯỚC 2: Đọc dữ liệu Dự đoán (Output) ---
    df_pred = pd.read_csv(FILE_PATH_PREDICTION)
    df_pred.columns = [c.lower().strip() for c in df_pred.columns]
    if 'close' not in df_pred.columns:
            df_pred['Close'] = df_pred.iloc[:, -1] 
    else:
            df_pred = df_pred.rename(columns={'close': 'Close'})

    # --- BƯỚC 3: TỰ ĐỘNG FILL NGÀY (Logic tự động điền ngày) ---
    if not df_hist.empty:
        last_hist_date = df_hist['Date'].iloc[-1]
        num_pred_days = len(df_pred)
        # Tạo ngày tương lai (Bỏ qua T7, CN)
        future_dates = pd.bdate_range(start=last_hist_date + BusinessDay(1), periods=num_pred_days, freq='B')
        df_pred['Date'] = future_dates
    else:
        df_pred['Date'] = pd.date_range(start=pd.Timestamp.now(), periods=len(df_pred))

    return df_hist, df_pred

# Load dữ liệu toàn cục
df_hist_global, df_pred_global = load_and_process_data()

# 2. HÀM VẼ BIỂU ĐỒ

def create_combined_chart(df_hist, df_pred):
    fig = go.Figure()

    # 1. Nến Lịch sử
    fig.add_trace(go.Candlestick(
        x=df_hist['Date'],
        open=df_hist['Open'], high=df_hist['High'],
        low=df_hist['Low'], close=df_hist['Close'],
        name='Lịch sử (OHLC)',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))

    # 2. Đường Dự đoán (Nối tiếp)
    if not df_hist.empty and not df_pred.empty:
        last_hist_pt = df_hist.iloc[[-1]][['Date', 'Close']]
        pred_data = df_pred[['Date', 'Close']]
        line_data = pd.concat([last_hist_pt, pred_data])

        fig.add_trace(go.Scatter(
            x=line_data['Date'],
            y=line_data['Close'],
            mode='lines+markers',
            name='Dự đoán AI',
            line=dict(color='#00e5ff', width=3), # Màu Cyan
            marker=dict(size=4)
        ))

        # Đường kẻ dọc phân chia
        split_date = df_hist['Date'].iloc[-1]
        fig.add_vline(x=split_date, line_width=1, line_dash="dash", line_color="white")
        
        fig.add_annotation(
            x=split_date, y=df_hist['High'].max(),
            text="HIỆN TẠI", showarrow=False, yshift=10,
            font=dict(color="white", size=10)
        )

    # 3. Ẩn ngày nghỉ (T7, CN)
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])] 
    )

    fig.update_layout(
        template='plotly_dark',
        title="Biểu đồ tổng hợp: Lịch sử & Dự báo",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        paper_bgcolor='rgba(20,20,20,1)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# 3. GIAO DIỆN DASH

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Mã CP:", className="fw-bold text-info"),
            # ID ở đây là 'ticker'
            dbc.Input(id="ticker", value="FPT", style={'font-weight': 'bold'}),
            # ID ở đây là 'msg'
            html.Div(id="msg", className="text-danger small mt-1")
        ], width=2, className="mt-4"),
        
        dbc.Col([
            html.H2("CONQ-007 - Stock Price Prediction", className="text-center mt-4 text-white"),
        ], width=8)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                # ID ở đây là 'main-chart'
                dbc.CardBody(dcc.Loading(dcc.Graph(id='main-chart'), type="graph"))
            ], color="dark", outline=True)
        ], width=12, className="mt-3")
    ])
], fluid=True, style={'min-height': '100vh', 'background-color': '#000'})

# 4. CALLBACK DUY NHẤT (ĐÃ SỬA LỖI)

@app.callback(
    [Output('main-chart', 'figure'), Output('msg', 'children')],
    [Input('ticker', 'value')]
)
def update_dashboard(ticker):
    # Kiểm tra mã cổ phiếu
    if not ticker or ticker.upper().strip() != 'FPT':
        empty = go.Figure()
        empty.update_layout(template='plotly_dark', title="KHÔNG CÓ DỮ LIỆU")
        return empty, "Hiện tại chưa hỗ trợ mã CP này."
    
    # Nếu đúng mã, vẽ biểu đồ
    fig = create_combined_chart(df_hist_global, df_pred_global)
    return fig, "" # Xóa thông báo lỗi

if __name__ == '__main__':
    app.run(debug=True)