import datajoint as dj
import numpy as np
import pickle
import os
import cv2
import math
dj.config['database.host'] = 'tutorial-db.datajoint.io'
dj.config['database.user'] = 'shulyalk'
dj.config['database.password'] = 'VY2eilxrw2121'
dj.conn()
schema = dj.schema('shulyalk_tutorial', locals())  # this might differ depending on how you setup

# Modification of local datafile- store dataset for every neuron
filename = 'C:/Users/sidso/Documents/DataJoint/Interview Project/ret1_data.pkl'
odata = np.load(filename, allow_pickle=True)
data_dir = "C:/Users/sidso/Documents/DataJoint/Interview Project/spikedata2/"
isExist = os.path.exists(data_dir)
if not isExist:
    os.mkdir(data_dir)
diter = 0  # keep track of total number of saved datasets for BaseData class assignment
sesVar = []
stimVar = []
nVar = []
for i in range(0, len(odata)):  # loop through all subjects in dataset
    for s in range(0, len(odata[i]['stimulations'])):  # loop through all stimulations
        for n in range(0, len(odata[i]['stimulations'][s]['spikes'])):  # loop through all neurons
            sesVar.append(i)
            stimVar.append(s)
            nVar.append(n)
            filename = "Dataset_" + str(diter) + ".pkl"  # save for each dataset with incrementing names
            a_file = open(data_dir + filename, "wb")
            # store all necessary values for each neuron data file
            spike_dict = {'session_date': odata[i]['session_date'], 'sample_number': odata[i]['sample_number'],
                          'subject_name': odata[i]['subject_name'], 'spikes': odata[i]['stimulations'][s]['spikes'][n],
                          'fps': odata[i]['stimulations'][s]['fps'], 'movie': odata[i]['stimulations'][s]['movie'],
                          'n_frames': odata[i]['stimulations'][s]['n_frames'], 'pixel_size':
                              odata[i]['stimulations'][s]['pixel_size'], 'stim_height':
                              odata[i]['stimulations'][s]['stim_height'], 'stim_width':
                              odata[i]['stimulations'][s]['stim_width'], 'stimulus_onset':
                              odata[i]['stimulations'][s]['stimulus_onset'], 'x_block_size':
                              odata[i]['stimulations'][s]['x_block_size'], 'y_block_size':
                              odata[i]['stimulations'][s]['y_block_size']}
            diter += 1  # increment across all subjects, stimulations, and neurons
            pickle.dump(spike_dict, a_file)
            a_file.close()


@schema
class BaseData(dj.Manual):
    # BaseSession class - manually allocate 177 data samples
    # Store session_count as primary key to index across all data dicts
    definition = """
    data_count: int          # data index(unique) 
    ---
    session_count: int          # session idx for each data point
    stimulation_count: int      # stim count for each data point
    neuron_count: int           # neuron idx for each data point
    """


# Manually add data to BaseData object
dlist = []
# Define vectors containing data count, session count, stimulation count, and neuron count across all
for d in range(0, diter - 1):  # loop through all datasets and store data_count across all
    addData = {
        'data_count': d,
        'session_count': sesVar[d],
        'stimulation_count': stimVar[d],
        'neuron_count': nVar[d]
    }
    dlist.append(addData)
base = BaseData()
base.insert(dlist, skip_duplicates=True)


@schema
class Session(dj.Imported):
    # Session Class: Define Session object containing session count, date, sample num, and subj name
    definition = """
    -> BaseData
    ---                           # none of these are unique across all sessions
    session_date: date            # session date 
    sample_number: int            # sample number
    subject_name: varchar(128)    # mouse name   
    """

    def _make_tuples(self, key):  # _make_tuples takes a single  argument `key`
        data_file = (data_dir + "Dataset_{data_count}.pkl").format(**key)
        data = np.load(data_file, allow_pickle=True)
        key['session_date'] = data['session_date']
        key['sample_number'] = data['sample_number']
        key['subject_name'] = data['subject_name']
        self.insert1(key)
        print('Populated Session for Dataset {data_count}'.format(**key))


session = Session()
session.populate()


@schema
class Stimulation(dj.Imported):
    # Load in Individual Stimulation Data
    definition = """
    -> Session
    ---
    fps: float              # The movie recording frequency in frames per second
    movie: longblob         # A numpy array containing the movie stimulus presented to the mouse retina. The array is
                            # shaped as (horizontal blocks, vertical blocks, frames)
    n_frames: int           # number of frames
    pixel_size: float       # pixel size of retina in um/pixel
    stim_height: int        # height of stimulus(movie) in pixels
    stim_width: int         # width of stimulus(movie) in pixels
    stimulus_onset: float   # onset of the stimulus from the beginning of the recording session in seconds
    x_block_size: int       # size of x (horizontal) blocks in pixels
    y_block_size: int       # size of y (vertical) blocks in pixels
    spike_train: longblob  # Stores array of spike times measured relative to beginning of session in 'external' store
    """

    def _make_tuples(self, key):  # _make_tuples takes a single argument `key`
        data_file = (data_dir + "Dataset_{data_count}.pkl").format(**key)
        data = np.load(data_file, allow_pickle=True)
        key['fps'] = data['fps']
        key['movie'] = data['movie']
        key['n_frames'] = data['n_frames']
        key['pixel_size'] = data['pixel_size']
        key['stim_height'] = data['stim_height']
        key['stim_width'] = data['stim_width']
        key['stimulus_onset'] = data['stimulus_onset']
        key['x_block_size'] = data['x_block_size']
        key['y_block_size'] = data['y_block_size']
        key['spike_train'] = data['spikes']
        self.insert1(key)
        print('Populated Stimulation for Dataset {data_count}'.format(**key))


stimulation = Stimulation()
stimulation.populate()


@schema
class DelayParam(dj.Lookup):
    # Define Delay Length Parameters for STA calculation
    definition = """
    dp_id: int      #unique id for spike detection parameter set
    ---
    delay: int        # delay length for STA calculation in ms  
    """


#  Manually add delays for STA calculation
# Test 5 different delay lengths
delay_p = DelayParam()
delay_p.insert1({'dp_id': 0, 'delay': 1}, skip_duplicates=True)
delay_p.insert1({'dp_id': 1, 'delay': 2}, skip_duplicates=True)
delay_p.insert1({'dp_id': 2, 'delay': 3}, skip_duplicates=True)
delay_p.insert1({'dp_id': 3, 'delay': 4}, skip_duplicates=True)
delay_p.insert1({'dp_id': 4, 'delay': 5}, skip_duplicates=True)


@schema
class STRFimages(dj.Computed):
    # Compute a Spike Triggered Average Image for variable amount of delays
    definition = """
    -> Stimulation
    -> DelayParam
    ---
    sta_image: longblob         # Nested lists containing sta image for each delay
    strf_image: longblob        # Nested lists containing pixel image scaled to retinal image
    """

    def _make_tuples(self, key):  # _make_tuples takes a single argument `key`
        # Compute STA for each neuron
        print('Populating for: ', key)
        sp_train = (Stimulation() & key).fetch('spike_train')  # fetch spike timings as 1d np array
        movie = (Stimulation() & key).fetch('movie')
        fps = (Stimulation() & key).fetch('fps')
        s_height = (Stimulation() & key).fetch('stim_height')
        s_width = (Stimulation() & key).fetch('stim_width')
        x_block_size = (Stimulation() & key).fetch('x_block_size')
        y_block_size = (Stimulation() & key).fetch('y_block_size')
        pixel_size = (Stimulation() & key).fetch('pixel_size')
        n_frames = (Stimulation() & key).fetch('n_frames')
        stimulus_onset = (Stimulation() & key).fetch('stimulus_onset')
        dl = (DelayParam() & key).fetch('delay')

        numSpikes = len(sp_train[0])
        sta_img = np.zeros(s_height[0], s_width[0])
        strf_img = np.zeros(s_height[0], s_width[0])
        pixel_mat = np.zeros((numSpikes, s_height[0], s_width[0]))
        st_cnt = 0
        for st in sp_train[0]:
            # print(st)
            # print(dl[0])
            stimulus_time = st - dl[0]  # calculate time of variable delay from beginning of session
            # print('Stim Time:' + str(stimulus_time[0]))
            if stimulus_time > stimulus_onset[0]:  # Only take frames after stimulus has been presented
                # Reshape horizontal/vertical block array into image with stim_width and stim_height
                # Scale image width by x_block_size and height by y_block_size
                dframe = stimulus_time * fps[0]  # convert time in seconds to frames
                # print(dframe[0])
                # print(fps[0])
                # print(np.shape(movie[0]))
                if math.floor(dframe[0]) < n_frames[0]:
                    img = movie[0][:, :, math.floor(dframe[0])]
                    dim = (s_width[0], s_height[0])
                    # print(dim)
                    # print(np.shape(img))
                    # print(x_block_size[0])
                    # print(y_block_size[0])
                    # Store image matrix rescaled based on x and y block sizes
                    pixel_mat[st_cnt, :, :] = cv2.resize(img, dim, fx=x_block_size[0], fy=y_block_size[0])
            st_cnt += 1
        # take image average across all spikes for each delay
        # print(np.shape(pixel_mat))
        # print(np.shape(np.mean(pixel_mat, axis=0)))
        sta_img = np.mean(pixel_mat, axis=0)            # take mean across all spikes
        strf_img = np.mean(pixel_mat, axis=0) * pixel_size  # scale to um (pixel_size=um/pixel)

        key['sta_image'] = sta_img
        key['strf_image'] = strf_img
        self.insert1(key)
        print('Computed STA and STRF for Dataset {data_count} and delay_id {dp_id}'.format(**key))
