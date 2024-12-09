import logging

import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Data Dictionary", page_icon="datarobot icon.svg", layout="wide"
)

st.image("datarobot logo.svg", width=200)
st.title("Dictionary")

# Add debug logging
logger.info("Data Dictionary page loaded")
logger.info(f"Session state keys: {st.session_state.keys()}")
logger.info(f"data_dictionaries content: {st.session_state.data_dictionaries}")

if not st.session_state.data_dictionaries:
    logger.warning("No data dictionaries found in session state")
    st.info(
        "Please upload and process data from the main page to view the data dictionary"
    )
else:
    logger.info(f"Found {len(st.session_state.data_dictionaries)} dictionaries")
    for name, dictionary in st.session_state.data_dictionaries.items():
        st.subheader(name)
        logger.info(f"Processing dictionary for {name}")

        try:
            # Convert dictionary to DataFrame
            dict_df = pd.DataFrame(dictionary)
            logger.info(f"Created DataFrame for {name} with shape {dict_df.shape}")

            # Make dictionary editable
            edited_df = st.data_editor(
                dict_df,
                use_container_width=True,
                num_rows="dynamic",
                key=f"dict_editor_{name}",
            )

            # Download button for dictionary
            csv = edited_df.to_csv(index=False)
            st.download_button(
                label="Download Data Dictionary",
                data=csv,
                file_name=f"{name}_dictionary.csv",
                mime="text/csv",
                key=f"download_dict_{name}",
            )

        except Exception as e:
            logger.error(
                f"Error processing dictionary for {name}: {str(e)}", exc_info=True
            )
            st.error(f"Error displaying dictionary for {name}: {str(e)}")

        st.markdown("---")

# Add helpful tips
with st.sidebar:
    st.markdown(
        """
    ### Using the Data Dictionary
    
    The data dictionary provides detailed information about each column in your datasets:
    
    - **Column**: The name of the column
    - **Data Type**: The type of data in the column
    - **Description**: A description of what the data represents
    
    You can:
    - Edit descriptions directly in the table
    - Download the dictionary as CSV
    - Use the information to better understand your data
    """
    )
