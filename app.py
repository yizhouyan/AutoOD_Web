from flask import Flask

from flask import Flask, request, render_template, flash, redirect, url_for, Response, send_file
from werkzeug.utils import secure_filename
import os
from loguru import logger
import time
from shutil import copyfile
from autood import run_autood, AutoODResults, get_default_detection_method_list,OutlierDetectionMethod

LOGGING_PATH = "static/job.log"
# configure logger
logger.add(LOGGING_PATH, format="{time} - {message}")

UPLOAD_FOLDER = 'files'
DOWNLOAD_FOLDER = 'results/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = True  # start debugging
app.secret_key = "super secret key"

ALLOWED_EXTENSIONS = {'arff', 'csv'}


@app.route('/', methods=['GET', 'POST'])
def home():
    return redirect('/autood/index')


@app.route('/autood/index', methods=['GET'])
def autood_form():
    return render_template('form.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def flask_logger():
    """creates logging information"""
    with open(LOGGING_PATH) as log_info:
        while True:
            data = log_info.read()
            yield data.encode()
            time.sleep(1)


@app.route("/running_logs", methods=["GET"])
def running_logs():
    """returns logging information"""
    return Response(flask_logger(), mimetype="text/plain", content_type="text/event-stream")

def get_detection_methods(methods):
    logger.info(f"selected methods = {methods}")
    name_to_method_map = {
        "lof": OutlierDetectionMethod.LOF,
        "knn": OutlierDetectionMethod.KNN,
        "if": OutlierDetectionMethod.IsolationForest,
        "mahala": OutlierDetectionMethod.Manalanobis
    }
    selected_methods = [name_to_method_map[method] for method in methods]
    return selected_methods

@app.route('/autood/index', methods=['POST'])
def autood_input():
    if 'file' not in request.files:
        flash('Please provide an input file or select a dataset.')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('Please provide an input file or select a dataset.')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # start autoOD computation
        index_col_name = request.form['indexColName']
        label_col_name = request.form['labelColName']
        outlier_range_min = float(request.form['outlierRangeMin'])
        outlier_range_max = float(request.form['outlierRangeMax'])
        detection_methods = get_detection_methods(request.form.getlist('detectionMethods'))
        if not detection_methods:
            flash('Please choose at least one detection method....')
            return redirect(request.url)
        results = call_autood(filename, outlier_range_min, outlier_range_max, detection_methods, index_col_name, label_col_name)
        if results.error_message:
            flash(results.error_message)
            return redirect(request.url)
        else:
            # Create empty job.log, old logging will be deleted
            final_log_filename = f"log_{file.filename.replace('.', '_')}_{int(time.time())}"
            copyfile(LOGGING_PATH, DOWNLOAD_FOLDER + final_log_filename)
            open(LOGGING_PATH, 'w').close()
            return render_template('result_summary.html', best_f1=results.best_unsupervised_f1_score,
                                   autood_f1=results.autood_f1_score, mv_f1=results.mv_f1_score,
                                   best_method=",".join(results.best_unsupervised_methods),
                                   final_results=results.results_file_name, training_log=final_log_filename)
    else:
        flash('File type is not supported.')
        return redirect(request.url)


@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = DOWNLOAD_FOLDER + filename
    return send_file(file_path, as_attachment=False, attachment_filename='')


def call_autood(filename, outlier_percentage_min, outlier_percentage_max, detection_methods, index_col_name, label_col_name):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    logger.info(f"Start calling autood with file {filename}...indexColName = {index_col_name}, labelColName = {label_col_name}")
    logger.info(
        f"Parameters: outlier_percentage_min = {outlier_percentage_min}%, outlier_percentage_max = {outlier_percentage_max}%")
    return run_autood(filepath, logger, outlier_percentage_min, outlier_percentage_max, detection_methods, index_col_name, label_col_name)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
