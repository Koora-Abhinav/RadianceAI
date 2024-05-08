from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import pickle
import seaborn as sns
app = Flask(__name__)

final_data=0
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('file')
    head_html = ""
    x=[]
    for uploaded_file in uploaded_files:
        if uploaded_file.filename != '':
            x.append(pd.read_csv(uploaded_file))
        else:
            return "No file uploaded"
    df=merge_files(x[0],x[1])
    global final_data
    final_data=df
    # head_html += "<h2>Head of " + uploaded_file.filename + "</h2>"
    head_html += df.head().to_html()
    return render_template('display.html', data=head_html)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Placeholder for prediction computation
    # Replace this with your actual prediction code

    prediction_result = "Placeholder for prediction result"
    global final_data
    prediction_result=predict(final_data)
    hist = sns.displot(prediction_result, bins=4)
    hist.set(xlabel='Solar Irradiance', ylabel='Count')
    hist.savefig('static/histogram.jpg')
    plt.clf()
    plt.cla()
    plt.figure(figsize=(15,8))
    line = sns.lineplot(x=final_data.date,y=prediction_result)
    line.set_xlabel('date')
    line.set_ylabel('Solar Irradiance')
    linefig=line.get_figure()
    linefig.savefig('static/lineplot.jpg')
    plot_pie_chart(final_data,prediction_result)
    return render_template('prediction.html', prediction_result=prediction_result)

@app.route('/llm', methods=['GET'])
def llm():
    # Placeholder for redirecting to LLM page
    return render_template('llm.html')



@app.route('/compute', methods=['POST'])
def compute():
    # Get the input text from the request
    input_text = request.json['inputText']
    
    # Perform computation (placeholder)
    # Replace this with your actual computation logic
    output_text = "Placeholder output for: " + input_text
    
    return jsonify(output_text)



def merge_files(df1,df2) -> pd.DataFrame:

  dates = []
  for i, row in df1.iterrows():
    year,month,day = map(str, [int(row['YEAR']),int(row['MO']),int(row['DY'])])
    dates.append(pd.to_datetime('-'.join([year,month,day])))
  df1['date'] = dates

  dates = []
  for i, row in df2.iterrows():
    year,month,day = map(str, [int(row['YEAR']),int(row['MO']),int(row['DY'])])
    dates.append(pd.to_datetime('-'.join([year,month,day])))
  df2['date'] = dates

  df1.drop(columns=['YEAR','MO','DY'], inplace = True)
  df2.drop(columns=['YEAR','MO','DY'], inplace = True)
  data = pd.merge(df1, df2, on='date')
  #data.drop(columns = ['CLRSKY_SFC_SW_DWN', 'date'], inplace=True)
  return data






def predict(data: pd.DataFrame) -> np.array:
    features = [
        'T2M_MAX',
        'T2M_RANGE',
        'RH2M',
        'PRECTOTCORR',
        'WS2M',
        'TS',
        'WS50M',
        'PS',
        'WD10M',
        'WD50M',
        'T2M_MIN',
        'T2M',
        'WS10M_RANGE',
        'WS50M_MAX',
        'WS10M_MAX',
        'WS50M_RANGE',
        'WS50M_MIN',
        'WS10M_MIN',
        'WS10M',
        'T2MWET',
        'ALLSKY_SFC_SW_DWN'
            ]
    print(data)
    data = data[features]

    target = 'ALLSKY_SFC_SW_DWN'
    x,y = data.drop(columns=[target]),data[target]
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
        x = scaler.transform(x)

    with open('best_svr_model.pkl','rb') as file:
        svr = pickle.load(file)
        y = svr.predict(x)
    
    print(y)
    return y

def plot_pie_chart(data,results):
    temp = pd.DataFrame({
        'month': list(map(lambda x: x.month, list(data.date))),
        'solar': list(results)
    })

    q1 = temp[temp.month<=3]
    q2 = temp[temp.month>=4][temp.month<=6]
    q3 = temp[temp.month>=7][temp.month<=9]
    q4 = temp[temp.month>=10][temp.month<=12]

    values = list(map(lambda x: x.solar.mean(), (q1,q2,q3,q4)))
    classes = [f'Q{i}' for i in range(1,5)]

    palette = sns.color_palette('bright')
    plt.clf()
    plt.cla()
    plt.figure()
    plt.pie(values,labels=classes,colors=palette,autopct='%0.f%%')
    plt.savefig('static/piechart.jpg')

    temp = pd.DataFrame({
        'date': list(data.date),
        'solar': list(results)
    })

    temp.to_csv("Prediction Results")
    

if __name__ == '__main__':
    app.run(debug=True)
