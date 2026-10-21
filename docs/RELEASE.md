# Credit Score release — 21 October 2026

The release candidate is ready for Streamlit Community Cloud: repository `SevenZ27/UAS`, branch `main`, entrypoint `app_streamlit.py`, Python 3.12, local CPU inference and no secrets. Deployment and public verification are being completed; this record will be updated with their evidence.

Model: `stage2-49c6876730d2`. SHA-256: `d8f6ce0302a8954f4f15be17ad726bd26068ffacb3b2396bc0867f73a0af20b1`. Frozen holdout report SHA-256: `3c38004eadfb556d0f27c6dc49f74bbb51a96fed2b42b91b5e2fca36e41a7cc9`.

Local verification: 119 tests passed; Ruff passed; independent frozen evaluation verification passed; fresh-process packaged hooks matched local predictions and warnings for three profiles. Browser checks at 1440, 900, 390 and 320 px passed normal/outside-range prediction, errors, reset, missing values, download and backend recovery, with zero axe violations in checked states. Keyboard, focus, 200% text, reduced motion and narrow confusion-table checks passed.

Evidence: [tests](verification/tests-release.xml), [browser](verification/browser-release-local.json), [keyboard](verification/keyboard-release-local.json), [evaluation](verification/evaluation-stage-two.json), [packaged hooks](verification/sagemaker-package.json). Historical milestone reports retain their recorded behavior; [SERVING-POLICY.md](SERVING-POLICY.md) defines the current app policy.

Both local ATS CVs have updated Credit Score entries and clickable application/source links. DOCX and PDF copies remain outside the public source repository. Each rendered PDF has two pages; unrelated project entries and original CV files are preserved.
