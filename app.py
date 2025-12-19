import streamlit as st
import io
import pandas as pd
from agents.dataset_intelligence import analyze_excel, inspect_uploaded_file
from analytics import metrics
from visuals import charts


def run_app():
    st.title("AI Market Study Agent (Base Version)")

    uploaded_files = st.file_uploader("Upload one or more CoStar Excel File(s)", accept_multiple_files=True, type=['xls', 'xlsx', 'csv'])

    if not uploaded_files:
        st.info("Upload one or more Excel/CSV files to begin analysis.")
        return

    for uploaded in uploaded_files:
        st.header(f"File: {uploaded.name}")
        # read as bytes and pass file-like to analyzer
        content = uploaded.read()
        bio = io.BytesIO(content)

        # File inspection
        inspection_result = inspect_uploaded_file(bio, uploaded.name)
        st.subheader("File Inspection Results")
        st.json(inspection_result)

        if not inspection_result['readable']:
            st.error(f"File could not be read: {inspection_result['error_message']}")
            continue

        result = analyze_excel(bio, source_file=uploaded.name)

        st.subheader("Detected Sheets and Normalized Data")
        for sheet in result.get('sheets', []):
            st.markdown(f"**Sheet:** {sheet.get('sheet_name')} — Type: {sheet.get('analysis_type')} — Aggregation: {sheet.get('aggregation')}")
            norm = pd.DataFrame(sheet.get('normalized', []))
            if norm.empty:
                st.write("No canonical data could be extracted from this sheet.")
                continue

            # Debugging output for normalized data
            st.write("Normalized Data Preview:")
            st.dataframe(norm.head())

            # show a trend chart for trend-like sheets
            if sheet.get('analysis_type') == 'trend':
                try:
                    fig = charts.plot_line_trend(norm, date_col='date', value_col='value', title=f"{uploaded.name} - {sheet.get('sheet_name')}")
                    st.pyplot(fig)
                    direction, slope = metrics.trend_direction(norm, date_col='date', value_col='value')
                    st.write({'trend_direction': direction, 'slope': slope})
                    yoy = metrics.calculate_yoy_growth(norm, date_col='date', value_col='value')
                    st.write('Latest YoY (last row):', float(yoy['yoy_growth'].dropna().tail(1)) if not yoy['yoy_growth'].dropna().empty else None)
                except Exception as e:
                    st.write('Could not plot trend:', str(e))

            # show stacked segment mix for segment-like sheets
            if sheet.get('analysis_type') == 'segment':
                try:
                    fig = charts.plot_stacked_segment(norm, date_col='date', segment_col='segment', value_col='value', title=f"{uploaded.name} - {sheet.get('sheet_name')}")
                    st.pyplot(fig)
                    shares = metrics.segment_share(norm, date_col='date', segment_col='segment', value_col='value')
                    st.write('Segment shares (sample):')
                    st.dataframe(shares.head(20))
                except Exception as e:
                    st.write('Could not plot segment chart:', str(e))

            # for growth sheets, show YoY series
            if sheet.get('analysis_type') == 'growth':
                try:
                    yoy = metrics.calculate_yoy_growth(norm, date_col='date', value_col='value')
                    st.line_chart(yoy.set_index('date')['yoy_growth'])
                except Exception as e:
                    st.write('Could not compute growth:', str(e))


if __name__ == "__main__":
    run_app()
