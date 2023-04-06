import pandas as pd
import streamlit as st 
from sklearn.impute import SimpleImputer
# from sqlalchemy import create_engine
# from urllib.parse import quote
import joblib, pickle
model = pickle.load(open('rf.pkl','rb'))
imp_enc_scale = joblib.load('imp_enc_scale')
outlier = joblib.load('winsorizer')


def predict(data):
    
    # engine = create_engine(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))
    SimpleImputer.get_feature_names_out = (lambda self, names=None:self.feature_names_in_)
    clean = pd.DataFrame(imp_enc_scale.transform(data), columns = imp_enc_scale.get_feature_names_out())
    clean[list(clean.iloc[:,0:5])] = outlier.transform(clean[list(clean.iloc[:,0:5])])
    
    prediction = pd.DataFrame(model.predict(clean), columns= ['is_Fraud'])
    
    final = pd.concat([prediction, clean], axis = 1)
    
    # final.to_sql('is_fraud_prediction', con = engine, if_exists = 'replace', )
    
    return final

def main():
    
    st.title("Anti_Money_laundring")
    st.sidebar.title("Anti_Money_laundring prediction")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Anti_Money_laundring </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("Upload the new data using CSV or Excel file.")
    
    
    # html_temp = """
    # <div style="background-color:tomato;padding:10px">
    # <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    # </div>
    # """
    # st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    # user = st.sidebar.text_input("user")
    # pw = st.sidebar.text_input("password", type = 'password')
    # db = st.sidebar.text_input("database", type = 'password')
    
    result = ""
    
    
    if st.button("Predict"):
        result = predict(data)
        
        # import seaborn as sns
        # cm = sns.light_palette("blue", as_cmap=True)
        # st.table(result.style.background_gradient(cmap=cm).set_precision(2))
        st.table(result)
    
if __name__=='__main__':
    main()



      
      