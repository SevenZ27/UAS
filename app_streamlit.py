"""Streamlit UI: grouped inputs, accessible category output and verified evidence."""

import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from credit_score.client import invoke_sagemaker
from credit_score.contract import BY_NAME, ContractError, PredictionError
from credit_score.presentation import (
    LABELS,
    category_markup,
    comparison_markup,
    confusion_markup,
    form_default,
    input_label,
    load_evidence,
    number,
    parse_form_inputs,
    snapshot_markup,
)
from credit_score.service import request_for, validate_client_result
from credit_score.serving_policy import load_exploration_service

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
ARTIFACTS = Path(os.environ.get("CREDIT_SCORE_ARTIFACTS", ROOT / "artifacts"))
st.set_page_config(page_title="Credit Score · Klasifikasi", layout="wide")
st.html(ROOT / "ui/styles.css")


@st.cache_resource
def get_service(path, fingerprint):
    return load_exploration_service(path)


def clear_result():
    had_result = "prediction" in st.session_state
    for key in ["prediction", "submitted_inputs", "result_token"]:
        st.session_state.pop(key, None)
    if had_result:
        st.session_state["result_notice"] = (
            "Input berubah. Jalankan prediksi lagi untuk melihat hasil terbaru."
        )


st.title("Klasifikasi kategori kredit")
st.html(
    '<a class="skip-link" href="#isi-profil">Lewati ke formulir</a><p class="intro">Jelajahi kategori Good, Standard, dan Poor dari profil numerik dataset. Isi profil, jalankan model, lalu baca hasil beserta batas penggunaannya.</p>'
)
st.html(
    '<p class="usage-note">Demonstrasi pembelajaran. Isi nominal mengikuti skala contoh; mata uang data aslinya belum diketahui dan tidak dikonversi ke rupiah. Hasil tidak menentukan persetujuan pinjaman.</p>'
)

try:
    fingerprint = (ARTIFACTS / "metadata.json").stat().st_mtime_ns
    service = get_service(str(ARTIFACTS), fingerprint)
except (PredictionError, OSError):
    st.error(
        "Model lokal belum siap. Jalankan build model sesuai README, lalu muat ulang halaman."
    )
    st.stop()

meta = service.metadata
with st.expander("Pengaturan sumber prediksi"):
    backend = st.radio(
        "Sumber prediksi",
        ["Lokal", "AWS SageMaker"],
        horizontal=True,
        key="backend",
        on_change=clear_result,
    )
    st.markdown(
        f"Model: `{meta['model_version']}` · Kontrak: `{meta['contract_version']}`"
    )
    st.write(
        "Mode Lokal menjalankan bundle di server aplikasi. Mode AWS memerlukan endpoint dengan versi model yang sama; input dikirim ke endpoint tersebut hanya saat prediksi dijalankan."
    )
    if (
        backend == "AWS SageMaker"
        and not os.environ.get("CREDIT_SCORE_ENDPOINT", "").strip()
    ):
        st.warning("Endpoint AWS belum diatur. Mode Lokal dapat digunakan sekarang.")


def reset_example():
    clear_result()
    st.session_state["result_notice"] = (
        "Contoh median training dimuat. Jalankan prediksi untuk melihat kategorinya."
    )
    for key, value in meta["defaults"].items():
        st.session_state["feature_" + key] = form_default(key, value)


groups = [
    (
        "Profil dan pendapatan",
        [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Num_of_Loan",
        ],
    ),
    (
        "Pinjaman dan riwayat",
        [
            "Interest_Rate",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Credit_History_Age",
        ],
    ),
    (
        "Pembayaran dan saldo",
        [
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Payment_of_Min_Amount",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
        ],
    ),
]
raw_inputs = {}
focus_target = None
focus_token = None
left, right = st.columns([1.65, 1], gap="large")
with left, st.container(key="input-panel"):
    st.header("Isi profil", anchor="isi-profil")
    st.write(
        "18 fitur dalam tiga bagian. Contoh menggabungkan median training, dibulatkan dua desimal untuk fitur nonhitungan. Ini bukan profil seorang nasabah."
    )
    st.button(
        "Muat contoh nilai dataset",
        on_click=reset_example,
        help="Mengembalikan seluruh isian ke median training dan menghapus hasil sebelumnya.",
    )
    with st.expander("Panduan satuan dan input kosong"):
        st.write(
            "Usia wajib diisi. Fitur lain boleh dikosongkan; model akan memakai median training dan menyebutkannya dalam peringatan hasil. Angka kosong tidak dianggap nol."
        )
        st.write(
            "Ketik angka tanpa pemisah ribuan; gunakan titik atau koma untuk desimal. Nilai di luar rentang contoh training tetap dapat diprediksi dengan peringatan. Nilai ekstrem bisa dibatasi oleh prapemrosesan model; hasil menjelaskan mana yang dibatasi."
        )
        st.write(
            "Gunakan satuan pada label. Tanda desimal titik dan koma sama-sama diterima. Contoh 1250,50 atau 1250.50; jangan tulis pemisah ribuan."
        )
    for title, names in groups:
        st.subheader(title)
        for i, name in enumerate(names):
            if i % 2 == 0:
                columns = st.columns(2)
            feature = BY_NAME[name]
            key = "feature_" + name
            if key not in st.session_state:
                value = meta["defaults"][name]
                st.session_state[key] = form_default(name, value)
            with columns[i % 2]:
                if name == "Payment_of_Min_Amount":
                    raw_inputs[name] = st.selectbox(
                        feature.label,
                        [0, 1, None],
                        format_func=lambda value: (
                            "Belum diketahui"
                            if value is None
                            else "Ya"
                            if value
                            else "Tidak"
                        ),
                        help=feature.help,
                        key=key,
                        on_change=clear_result,
                    )
                else:
                    bounds = meta["supported_ranges"][name]
                    hint = (
                        (
                            "Wajib diisi. "
                            if name == "Age"
                            else "Boleh kosong: diimputasi median training. "
                        )
                        + f"Rentang contoh training {number(bounds['min'], 0 if feature.integer else 2)}–{number(bounds['max'], 0 if feature.integer else 2)}; di luar rentang tetap diproses dengan peringatan. "
                        + feature.help
                    )
                    raw_inputs[name] = st.text_input(
                        input_label(feature),
                        help=("Bilangan bulat. " if feature.integer else "") + hint,
                        key=key,
                        on_change=clear_result,
                        placeholder="Wajib diisi"
                        if name == "Age"
                        else "Kosong = median training",
                    )
    st.write(
        "Usia wajib diisi. Di luar rentang contoh tetap mendapat prediksi dengan catatan. Perubahan isian menghapus hasil sebelumnya."
    )
    if st.button("Prediksi kategori", type="primary", width="stretch"):
        clear_result()
        focus_token = uuid4().hex
        try:
            with st.spinner("Memproses profil…"):
                inputs, parse_issues = parse_form_inputs(
                    raw_inputs, meta["supported_ranges"]
                )
                if parse_issues:
                    raise ContractError(parse_issues)
                payload = request_for(inputs)
                if backend == "Lokal":
                    result = service.predict(payload)
                else:
                    result = validate_client_result(invoke_sagemaker(payload))
                    if result["model_version"] != meta["model_version"]:
                        raise PredictionError(
                            "Versi model AWS berbeda dari konfigurasi input. Muat bundle yang sesuai."
                        )
            st.session_state.prediction = result
            st.session_state.submitted_inputs = dict(inputs)
            st.session_state.result_token = focus_token
            focus_target = "hasil-prediksi"
        except ContractError as exc:
            st.html('<h3 id="prediction-error">Periksa isian profil</h3>')
            for issue in exc.issues:
                label = (
                    BY_NAME[issue["field"]].label
                    if issue.get("field") in BY_NAME
                    else "Input"
                )
                st.error(f"{label}: {issue.get('message', 'Nilai tidak valid.')}")
            focus_target = "prediction-error"
        except PredictionError as exc:
            st.html('<h3 id="prediction-error">Prediksi belum tersedia</h3>')
            st.error(str(exc))
            focus_target = "prediction-error"
        except Exception:
            LOGGER.exception("Unexpected Credit Score prediction failure")
            st.html('<h3 id="prediction-error">Prediksi belum tersedia</h3>')
            st.error(
                "Prediksi belum dapat diproses. Muat ulang halaman lalu coba lagi."
            )
            focus_target = "prediction-error"

with right, st.container(key="results-panel"):
    st.header("Hasil prediksi", anchor="hasil-prediksi")
    if "prediction" not in st.session_state:
        st.write("**Belum ada prediksi**")
        st.write(
            st.session_state.get(
                "result_notice",
                "Gunakan contoh yang tersedia atau sesuaikan profil, lalu pilih Prediksi kategori.",
            )
        )
        st.html(category_markup())
    else:
        result = st.session_state.prediction
        label = result["labels"][0]
        st.write(f"**Kategori model: {LABELS[label]} ({label})**")
        st.html(category_markup(label))
        st.html(
            '<p class="result-note">Penanda menunjukkan kategori keluaran model, bukan posisi pada skala skor atau probabilitas gagal bayar.</p>'
        )
        st.write(f"Hasil dari {backend}. Mengubah input akan menghapus hasil ini.")
        range_details = []
        for message in result["warnings"]:
            if "rentang contoh training" in message:
                range_details.append(message)
            else:
                st.warning(message)
        if range_details:
            with st.expander("Rincian nilai di luar rentang"):
                for message in range_details:
                    st.write(message)
        with st.expander("Input yang dipakai dan versi model"):
            st.markdown(f"Versi model: `{result['model_version']}`")
            st.html(snapshot_markup(st.session_state.submitted_inputs))
        download = {
            "prediction": result,
            "inputs": st.session_state.submitted_inputs,
            "meaning": "Dataset category demonstration; not a loan approval or default probability. Out-of-training-range inputs may be clipped and carry warnings.",
        }
        st.download_button(
            "Unduh ringkasan JSON",
            data=json.dumps(download, indent=2, ensure_ascii=False),
            file_name="credit-score-result.json",
            mime="application/json",
            on_click="ignore",
            width="stretch",
        )
    st.divider()
    st.write("**Cara membaca kategori**")
    st.write(
        "Good, Standard, dan Poor mengikuti label dataset. Kategori ini tidak menyatakan tingkat keamanan finansial atau keputusan persetujuan kredit."
    )
    st.write(
        "Probabilitas individual dan faktor penyebab tidak ditampilkan karena belum menjadi keluaran yang tervalidasi pada aplikasi ini."
    )

st.divider()
st.header("Evaluasi model", anchor="evaluasi-model")
st.write(
    "Konteks untuk menilai model secara keseluruhan, bukan tingkat keyakinan profil yang sedang kamu isi."
)
try:
    report = load_evidence(ROOT, meta)
except (OSError, ValueError, KeyError):
    report = None
if report is None:
    st.warning(
        "Laporan evaluasi yang cocok dengan bundle aktif belum tersedia. Prediksi lokal tetap dapat digunakan; angka dari versi lain tidak ditampilkan."
    )
else:
    with st.expander("Lihat benchmark dan kesalahan per kelas"):
        st.subheader("Accuracy pada nasabah holdout")
        st.write(
            f"{number(report['holdout_rows'], 0)} observasi dari {number(report['holdout_customers'], 0)} nasabah, terpisah dari training dan pemilihan model."
        )
        st.html(comparison_markup(report))
        lo, hi = report["bootstrap"]["intervals"]["accuracy"]
        st.write(
            f"Interval bootstrap nasabah 95%: {number(lo * 100)}–{number(hi * 100)}%. Macro-F1 model {number(report['selected']['macro_f1'], 3)}; baseline {number(report['prior_baseline']['macro_f1'], 3)}."
        )
        coverage = report["serving_domain"]
        st.write(
            f"Sebanyak {number(coverage['accepted'], 0)} dari {number(coverage['total'], 0)} profil holdout ({number(coverage['coverage'] * 100, 1)}%) berada dalam domain ketat versi evaluasi awal. Versi aplikasi kini juga memproses nilai di luar rentang contoh dengan peringatan. Benchmark di atas memakai seluruh holdout melalui pipeline; ini bukan jaminan akurasi profil individual."
        )
        st.subheader("Prediksi yang masih keliru")
        st.html(confusion_markup(report))
        st.write(
            "Baris menunjukkan kelas aktual; kolom menunjukkan prediksi. Sel diagonal adalah prediksi yang cocok. Persentase dihitung terhadap total tiap baris."
        )
        st.write(
            "Precision Good masih 54,84%. Pada kelompok usia 45+, recall Poor hanya 30,16% dari 126 observasi berlabel Poor. Temuan kelompok ini bersifat deskriptif, belum merupakan validasi fairness."
        )
        st.write(
            "Lima fold development memisahkan nasabah; enam kandidat dibandingkan sebelum holdout dinilai. Interval bersyarat pada model tetap dan dataset ini, tidak mencakup ketidakpastian pelatihan ulang atau perubahan populasi."
        )
with st.expander("Tentang input dan model"):
    st.write(
        "18 fitur memakai satu pipeline parsing, imputasi median training, pembatasan outlier IQR training, dan Random Forest. Fitur hitungan tetap bilangan bulat; perubahan limit dan selisih jatuh tempo dapat bernilai negatif."
    )
    st.write(
        "Mata uang, pembentukan label, dan provenance sumber belum terverifikasi. Evaluasi ini menguji nasabah terpisah pada dataset yang sama, belum data eksternal atau periode mendatang."
    )
    st.markdown(
        f"Bundle aktif: `{meta['model_version']}` · Kontrak `{meta['contract_version']}`"
    )
if focus_target:
    st.html(
        f'<span class="sr-only" data-focus-request="{focus_token}" data-focus-target="{focus_target}"></span>'
    )

st.html(
    "<script>"
    + (ROOT / "ui/accessibility.js").read_text(encoding="utf8")
    + "</script>",
    unsafe_allow_javascript=True,
)
