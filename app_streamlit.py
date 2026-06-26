import json
import os
import boto3
import streamlit as st

from botocore.exceptions import (
    ClientError,
    NoCredentialsError
)

ENDPOINT_NAME = os.environ.get(
    "ENDPOINT_NAME",
    "UAS-endpoint"
)

REGION = os.environ.get(
    "AWS_REGION",
    "us-east-1"
)


@st.cache_resource
def get_runtime_client():
    return boto3.client(
        "sagemaker-runtime",
        region_name=REGION
    )


def invoke_endpoint(features):
    runtime = get_runtime_client()

    payload = {
        "instances": [features]
    }

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )

    return json.loads(
        response["Body"].read().decode("utf-8")
    )


# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Skor Kredit",
    layout="wide"
)

# Judul Utama
st.title("Sistem Klasifikasi Skor Kredit")

st.markdown("""
Masukkan informasi keuangan nasabah di bawah ini untuk 
memprediksi kategori **Skor Kredit (Credit Score)** mereka secara otomatis.
""")

# Form Input Data
col1, col2 = st.columns(2)

with col1:
    st.subheader("Profil & Kredit Dasar")
    age = st.number_input("Usia (Tahun)", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Pendapatan Tahunan", min_value=0.0, value=120000000.0)
    monthly_salary = st.number_input("Gaji Bersih Bulanan (Take Home Pay)", min_value=0.0, value=10000000.0)
    bank_accounts = st.number_input("Jumlah Rekening Bank Aktif", min_value=0, value=2)
    credit_cards = st.number_input("Jumlah Kartu Kredit yang Dimiliki", min_value=0, value=1)
    interest_rate = st.number_input("Suku Bunga Rata-rata (%)", min_value=0.0, value=12.5)
    num_loans = st.number_input("Jumlah Pinjaman Aktif (Loan)", min_value=0, value=1)
    delay_due = st.number_input("Rata-rata Keterlambatan Jatuh Tempo (Hari)", min_value=0, value=2)
    delayed_payment = st.number_input("Frekuensi Telat Bayar (Bulan Terakhir)", min_value=0, value=1)

with col2:
    st.subheader("Riwayat & Perilaku Keuangan")
    changed_limit = st.number_input("Persentase Perubahan Batas Kredit (%)", value=0.0)
    inquiries = st.number_input("Jumlah Pengajuan Kredit Baru (Inquiries)", min_value=0, value=1)
    outstanding_debt = st.number_input("Sisa Total Utang (Outstanding Debt)", min_value=0.0, value=5000000.0)
    utilization_ratio = st.number_input("Rasio Penggunaan Kredit / Utilitas (%)", min_value=0.0, value=25.0)
    credit_history_age = st.number_input("Usia Riwayat Kredit (Dalam Bulan)", min_value=0, value=24)
    payment_min = st.selectbox("Sering Membayar Jumlah Minimum Saja?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    total_emi = st.number_input("Total Cicilan Bulanan (EMI)", min_value=0.0, value=1500000.0)
    invested_monthly = st.number_input("Nominal Investasi Bulanan", min_value=0.0, value=1000000.0)
    monthly_balance = st.number_input("Sisa Saldo Tabungan Bulanan", value=2500000.0)

# Tombol Prediksi
if st.button(
    "Analisis dan Prediksi Skor Kredit",
    use_container_width=True
):
    user_data = {
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": bank_accounts,
        "Num_Credit_Card": credit_cards,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_loans,
        "Delay_from_due_date": delay_due,
        "Num_of_Delayed_Payment": delayed_payment,
        "Changed_Credit_Limit": changed_limit,
        "Num_Credit_Inquiries": inquiries,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": utilization_ratio,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_min,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invested_monthly,
        "Monthly_Balance": monthly_balance
    }

    try:
        result = invoke_endpoint(user_data)

        # Expecting same response structure as original endpoint (labels)
        label = result.get("labels", [None])[0]

        st.divider()
        st.subheader("Hasil Analisis:")

        if label == "Good":
            st.success(f"**Hasil: BAIK (Good)** — Nasabah memiliki kredibilitas keuangan yang sangat aman.")
        elif label == "Standard":
            st.warning(f"**Hasil: STANDAR (Standard)** — Nasabah berada di zona aman, namun dengan catatan tertentu.")
        else:
            st.error(f"**Hasil: BURUK (Bad)** — Nasabah berisiko tinggi. Pertimbangkan kembali pemberian kredit.")

    except NoCredentialsError:
        st.error("AWS credentials not found.")

    except ClientError as e:
        st.error(f"AWS Error: {e}")