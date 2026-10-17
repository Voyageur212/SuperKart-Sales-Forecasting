"""SuperKart sales prediction UI | Author: Dimas Pradhana."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='SuperKart Sales Forecasting', layout='wide')
st.title('SuperKart | Sales Forecasting')
st.caption('Project author: Dimas Pradhana | Model: tuned Random Forest')
MODEL_PATH = Path(__file__).parent / 'superkart_model.joblib'
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
if not MODEL_PATH.exists():
    st.error('Missing superkart_model.joblib. Upload the trained model to the repository root.')
    st.stop()
model = load_model()
FEATURES = list(model.feature_names_in_)
PERISHABLE = {'Meat','Dairy','Bread','Breakfast','Frozen Foods','Fruits and Vegetables','Seafood'}
def prepare(df):
    df=df.copy()
    if 'Product_Id_char' not in df and 'Product_Id' in df:
        df['Product_Id_char']=df['Product_Id'].astype(str).str[:2]
    if 'Store_Age_Years' not in df and 'Store_Establishment_Year' in df:
        df['Store_Age_Years']=2026-pd.to_numeric(df['Store_Establishment_Year'],errors='coerce')
    if 'Product_Type_Category' not in df and 'Product_Type' in df:
        df['Product_Type_Category']=df['Product_Type'].apply(lambda x:'Perishables' if x in PERISHABLE else 'Non Perishables')
    missing=[x for x in FEATURES if x not in df]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    return df[FEATURES]

t1,t2=st.tabs(['Single prediction','Batch CSV predictions'])
with t1:
    with st.form('single'):
        a,b,c=st.columns(3)
        with a:
            weight=st.number_input('Product weight',min_value=0.,value=12.66)
            sugar=st.selectbox('Sugar content',['Low Sugar','Regular','No Sugar'])
            area=st.number_input('Allocated area ratio',min_value=0.,value=.05,format='%.4f')
            mrp=st.number_input('Product MRP',min_value=0.,value=140.)
        with b:
            size=st.selectbox('Store size',['Small','Medium','High'])
            city=st.selectbox('City tier',['Tier 1','Tier 2','Tier 3'])
            store=st.selectbox('Store type',['Supermarket Type1','Supermarket Type2','Departmental Store','Food Mart'])
        with c:
            pid=st.text_input('Product ID',value='FDX01')
            year=st.number_input('Store establishment year',min_value=1900,max_value=2026,value=2005)
            ptype=st.selectbox('Product type',['Dairy','Soft Drinks','Meat','Fruits and Vegetables','Snack Foods','Household','Others','Bread','Frozen Foods','Canned','Baking Goods','Health and Hygiene','Hard Drinks','Breakfast','Seafood','Starchy Foods'])
        submit=st.form_submit_button('Predict sales')
    if submit:
        row=pd.DataFrame([{'Product_Weight':weight,'Product_Sugar_Content':sugar,'Product_Allocated_Area':area,'Product_MRP':mrp,'Store_Size':size,'Store_Location_City_Type':city,'Store_Type':store,'Product_Id':pid,'Store_Establishment_Year':year,'Product_Type':ptype}])
        try: st.metric('Predicted product-store sales',f'{float(model.predict(prepare(row))[0]):,.2f}')
        except Exception as e: st.error(str(e))
with t2:
    uploaded=st.file_uploader('Upload SuperKart batch CSV',type='csv')
    if uploaded:
        try:
            batch=pd.read_csv(uploaded)
            preds=model.predict(prepare(batch))
            result=batch.copy();result['Predicted_Product_Store_Sales_Total']=preds
            st.success(f'Predicted {len(result):,} rows successfully')
            st.dataframe(result.head(100),use_container_width=True)
            st.download_button('Download predictions CSV',result.to_csv(index=False).encode('utf-8'),'superkart_predictions.csv','text/csv')
        except Exception as e: st.error(str(e))
st.info('These are product-store sales predictions. A calendar-quarter forecast requires dated sales history and aggregation assumptions.')
