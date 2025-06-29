import numpy as np
import cv2 as cv
import pandas as pd
import sys, os, logging

logging.basicConfig(filename='Results/runlog.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger("PIL.TiffImagePlugin").disabled=True

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

logging.info('CHECKING ACCURACY')


def get_mean_dist(m3id):
    if not os.path.isdir(f'Results/Worked/{m3id}'):
        logging.error(f"{m3id} failed")
        return -1
    matches = np.load(f'Results/Worked/{m3id}/{m3id}_MATCHES.npy')
    H, mask = cv.findHomography(matches[0], matches[1], cv.RANSAC, 3)
    if H is None:
        return -1
    
    hpoints_in = matches[0][mask.ravel() == 1]
    hpoints_out = matches[1][mask.ravel() == 1]

    homogeneous_points = np.hstack((hpoints_in, np.ones((len(hpoints_in), 1))))
    transformed_points_homogeneous = H @ homogeneous_points.T
    hpoints_in_tf = (transformed_points_homogeneous[:2, :].T / transformed_points_homogeneous[2, :].reshape(-1, 1))

    d_tf = np.linalg.norm(hpoints_in_tf - hpoints_out, axis=1)
    mean_dist = np.mean(d_tf)
    if np.isnan(mean_dist):
        return -1
    return mean_dist


def show_results():
    acc_df = pd.read_csv('Results/accuracy.csv')
    md = acc_df['MEAN_DIST'].values
    md = md[md>0]
    logging.info(f"Min: {np.min(md)}, Max: {np.max(md)}, Mean: {np.mean(md)}")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == 'show':
            show_results()
        else:
            logging.info(f"Mean Dist for {sys.argv[1]}: {get_mean_dist(sys.argv[1])}")
        
    elif len(sys.argv) > 1 and sys.argv[1] == '-f':

        fnout = 'accuracy.csv' if len(sys.argv) < 4 else sys.argv[3]
        m3ids = open(sys.argv[2], 'r').read().split('\n')

        if fnout not in os.listdir('Results'):
            data = pd.DataFrame(columns=['M3ID', 'MEAN_DIST'])
        else:
            data = pd.read_csv(f'Results/{fnout}')
        
        for k_m3id in range(len(m3ids)):
            logging.info(f"Starting {m3ids[k_m3id]}")
            mdist = get_mean_dist(m3ids[k_m3id])
            k_data = {'M3ID': m3ids[k_m3id], 'MEAN_DIST': mdist}
            data = pd.concat([data, pd.DataFrame(k_data, index=[0])])
            data.to_csv(f'Results/{fnout}',index=False)
        show_results()
    else:
        logging.error("INVALID PARAMS\nSINGLE RUN:\t python LTB_FM_M3.py <M3ID>\nBATCH RUN:\t python LTB_FM_M3.py -f <M3 list file>")
    