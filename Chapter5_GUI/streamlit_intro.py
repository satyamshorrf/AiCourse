# 1. Add the streamlit library to the project.
import streamlit as st
# 2. Add the title to the app.
st.title('My Awesome Streamlit App')

# 3. Running the App
# open the local terminal
# cd into the directory where this file is located: cd Chapter_GUI
# and run the following command: streamlit run streamlit_intro.py


# 4. Header
st.header('This is a header')

# 5. Subheader
st.subheader('This is a subheader')

# 6. Text
st.text('This is a text')

# 7. Markdown
st.markdown('This is a markdown')
st.markdown('## This is a Header')

# 8. Button
st.button('Click Me')

# 9. Checkbox
st.radio('Option A', ('Option A', 'Option B'))

# 10.Radio button
st.radio('Radio', ['Option 1', 'Option 2', 'Option 3'])

# 11. Selectbox
st.selectbox('Select', ['Option 1', 'Option 2', 'Option 3'])

# 12. File Uploader
st.file_uploader('File uploader', type=['png', 'jpg'])

# 13. Color Picker
st.color_picker('Color picker')

# 14. Date Input
st.date_input('Date input')

# 15. Time Input
st.time_input('Time input')

# 16. Text Input
st.text_input('Text input', placeholder='Enter your Name')

# 17. Number Input
st.number_input('Number input', min_value=1, max_value=100, value=50)

# 18. Text Area
st.text_area('Text area', placeholder='Enter your message here')

# 19. Slider
st.slider('Slider', min_value=0, max_value=100, value=50)

# 20. Progress Bar
# import time
# my_bar = st.progress(0)
# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)
#
# # 21. Spinner
# with st.spinner('Waiting...'):
#     time.sleep(2)

# 22. Adding Columns
col1, col2 = st.columns(2)

with col1:
    st.header('Column 1')
    st.text('Some text in column 1')

with col2:  # second column
    st.header('Column 2')
    st.text('Some text in column 2')

# 23. Add Image from file uploader and display it
image = st.file_uploader('Upload a file', type=['png', 'jpg', 'jpeg'])
if image:
    st.image(image, caption='Uploaded Image', use_column_width=True)


