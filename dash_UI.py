import dash
from dash import dcc, html, Input, Output, State, DiskcacheManager, CeleryManager
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import os
import diskcache 
import importlib
import model_train


# 1. CẤU HÌNH BACKGROUND CALLBACK MANAGER
cache = diskcache.Cache("./cache_directory")
background_callback_manager = DiskcacheManager(cache)

# Cấu hình đường dẫn file
FILE_PATH_HISTORY = '' # Đường dẫn input
FILE_PATH_PREDICTION = '' # Đường dẫn output 


# 2. XỬ LÝ DỮ LIỆU & VẼ BIỂU ĐỒ (Giữ nguyên logic cũ)
def load_data_strict():
    if not os.path.exists(FILE_PATH_HISTORY):
        return None, None, f"LỖI: Không tìm thấy file input tại {FILE_PATH_HISTORY}"

    try:
        # --- Đọc Input ---
        df_hist = pd.read_csv(FILE_PATH_HISTORY)
        df_hist.columns = [c.lower().strip() for c in df_hist.columns]
        if 'time' in df_hist.columns: df_hist = df_hist.rename(columns={'time': 'Date'})
        rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
        df_hist = df_hist.rename(columns=rename_map)
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        df_hist = df_hist.sort_values('Date')

        # --- Đọc Output (Nếu chưa có file dự đoán thì trả về None ở phần này) ---
        if os.path.exists(FILE_PATH_PREDICTION):
            df_pred = pd.read_csv(FILE_PATH_PREDICTION)
            df_pred.columns = [c.lower().strip() for c in df_pred.columns]
            if 'close' not in df_pred.columns: df_pred['Close'] = df_pred.iloc[:, -1] 
            else: df_pred = df_pred.rename(columns={'close': 'Close'})
            
            # Fill ngày
            last_hist_date = df_hist['Date'].iloc[-1]
            future_dates = pd.bdate_range(start=last_hist_date + BusinessDay(1), periods=len(df_pred), freq='B')
            df_pred['Date'] = future_dates
        else:
            df_pred = pd.DataFrame() # Trả về rỗng nếu chưa train xong

        return df_hist, df_pred, "" 

    except Exception as e:
        return None, None, f"LỖI DỮ LIỆU: {str(e)}"

def create_chart(df_hist, df_pred):
    fig = go.Figure()

    # Nến Lịch sử
    fig.add_trace(go.Candlestick(
        x=df_hist['Date'], open=df_hist['Open'], high=df_hist['High'],
        low=df_hist['Low'], close=df_hist['Close'],
        name='Lịch sử'
    ))

    # Đường Dự đoán (Chỉ vẽ nếu có dữ liệu)
    if not df_pred.empty:
        last_hist_pt = df_hist.iloc[[-1]][['Date', 'Close']]
        pred_data = df_pred[['Date', 'Close']]
        line_data = pd.concat([last_hist_pt, pred_data])

        fig.add_trace(go.Scatter(
            x=line_data['Date'], y=line_data['Close'],
            mode='lines+markers', name='Dự đoán',
            line=dict(color='#00e5ff', width=2), marker=dict(size=3)
        ))
        
        # Vạch ngăn cách
        split_date = df_hist['Date'].iloc[-1]
        fig.add_vline(x=split_date, line_dash="dash", line_color="white")

    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        rangeselector=dict(
            buttons=list([
                dict(count=14, label="14D", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="#333", font=dict(color="white"), activecolor="#00e5ff", y=1.05
        )
    )
    fig.update_layout(
        template='plotly_dark', height=600, dragmode='pan',
        margin=dict(l=30, r=30, t=80, b=30),
        legend=dict(orientation="h", y=1, x=0), hovermode='x unified'
    )
    return fig


# 3. GIAO DIỆN DASH
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], background_callback_manager=background_callback_manager)

app.layout = dbc.Container([
    # --- HEADER & CONTROLS ---
    dbc.Row([
        dbc.Col([
            html.Label("Mã CP:", className="fw-bold text-info"),
            dbc.Input(id="ticker", value="FPT", style={'font-weight': 'bold'}),
        ], width=2, className="mt-4"),
        
        dbc.Col([
            html.H2("CONQ007 - Stock Price Prediction", className="text-center mt-4 text-white"),
        ], width=5), # Giảm width từ 6 xuống 5 để nhường chỗ cho nút Clear

        # Khu vực nút bấm (Train + Clear)
        dbc.Col([
            dbc.Row([
                # Nút Train
                dbc.Col(dbc.Button("TRAIN MODEL", id="btn-train", color="danger", className="w-100 fw-bold"), width=8),
                # Nút Clear (Mới)
                dbc.Col(dbc.Button("CLEAR", id="btn-clear", color="secondary", className="w-100 fw-bold"), width=4),
            ]),
            
            # Thanh tiến trình (Progress Bar)
            html.Div([
                dbc.Progress(id="train-progress", value=0, striped=True, animated=True, className="mt-2", style={"height": "20px", "display": "none"}),
                html.Div(id="train-status", className="text-muted small mt-1 text-center")
            ])
        ], width=5, className="mt-4"), # Tăng width từ 4 lên 5
    ]),

    dbc.Row([
        dbc.Col(html.Label("Độ dài Dự đoán (Ngày):", className="text-warning fw-bold"), width=12),
        dbc.Col([
            dcc.RadioItems(
                id='pred-length-selector',
                options=[
                    {'label': '5 Ngày', 'value': 5},
                    {'label': '15 Ngày', 'value': 15},
                    {'label': '25 Ngày', 'value': 25},
                    {'label': '50 Ngày', 'value': 50},
                    {'label': '75 Ngày', 'value': 75},
                    {'label': '100 Ngày (Max)', 'value': 100}
                ],
                value=100,  # Mặc định hiển thị 100 ngày
                inline=True,
                className="mt-1",
                labelStyle={'paddingRight': '15px', 'color': 'white'}
            )
        ], width=12),
    ], className="mb-4"),

    # --- ERROR MESSAGE ---
    dbc.Row(dbc.Col(html.Div(id="error-msg", className="text-danger fw-bold text-center mt-2"), width=12)),

    # --- MAIN CHART ---
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody(dcc.Loading(dcc.Graph(id='main-chart', config={'scrollZoom': True}), type="graph")),
                color="dark", outline=True
            )
        ], width=12, className="mt-3")
    ])
], fluid=True, style={'min-height': '100vh', 'background-color': '#000'})


# 4. CALLBACK BACKGROUND (XỬ LÝ TRAIN MODEL)
@app.callback(
    output=Output("train-status", "children"),
    inputs=Input("btn-train", "n_clicks"),
    background=True,
    running=[
        (Output("btn-train", "disabled"), True, False), # Disable nút Train khi đang chạy
        (Output("btn-clear", "disabled"), True, False), # Disable nút Clear khi đang chạy
        (Output("train-progress", "style"), {"display": "flex"}, {"display": "none"}), 
        (Output("train-progress", "value"), 100, 0), 
    ],
    prevent_initial_call=True
)
def run_training_process(n_clicks):
    import time
    if not model_train:
        return "Lỗi: Chưa tìm thấy file model_train.py"
    
    start_time = time.time()
    try:
        # GỌI HÀM MAIN() TỪ FILE MODEL CỦA BẠN
        print(">>> UI: Bắt đầu gọi lệnh Train...")
        model_train.main() 
        
        duration = round(time.time() - start_time, 2)
        return f"Train hoàn tất trong {duration}s. Dữ liệu đã cập nhật."
    except Exception as e:
        return f"Lỗi khi train: {str(e)}"


# 5. CALLBACK CẬP NHẬT BIỂU ĐỒ (KHI TRAIN XONG, ĐỔI MÃ, HOẶC BẤM CLEAR)
@app.callback(
    [Output('main-chart', 'figure'), Output('error-msg', 'children')],
    [Input('ticker', 'value'), Input('train-status', 'children'), Input('btn-clear', 'n_clicks')]
)
def update_chart(ticker, train_status, btn_clear, selected_pred_length):
    # Xác định nút nào vừa được bấm
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger_id = 'No triggers'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # TRƯỜNG HỢP BẤM CLEAR
    if trigger_id == 'btn-clear':
        empty = go.Figure()
        empty.update_layout(
            template='plotly_dark', 
            title="Dữ liệu đã được xóa",
            xaxis={"visible": False}, 
            yaxis={"visible": False}
        )
        return empty, "Đã xóa dữ liệu."

    # TRƯỜNG HỢP LOAD DỮ LIỆU BÌNH THƯỜNG
    # Check mã
    if not ticker or ticker.upper().strip() != 'FPT':
        empty = go.Figure()
        empty.update_layout(template='plotly_dark', title="Không có dữ liệu")
        return empty, "Hiện tại chỉ hỗ trợ mã FPT"

    # Load lại dữ liệu
    df_h, df_p, err_msg = load_data_strict()
    
    if err_msg:
        empty = go.Figure()
        empty.update_layout(template='plotly_dark', title="NO DATA")
        return empty, err_msg
    
    if selected_pred_length and not df_p.empty:
        # Lấy số ngày dự đoán mong muốn
        N = int(selected_pred_length) 
        
        # Cắt DataFrame dự đoán theo số ngày N
        df_p_sliced = df_p.head(N).copy() 
    else:
        df_p_sliced = df_p.copy()
    
    return create_chart(df_h, df_p_sliced), ""

if __name__ == '__main__':
    app.run(debug=True)