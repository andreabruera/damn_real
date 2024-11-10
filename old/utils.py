import argparse
import os

def prepare_input_output_folders(args, mode='damaging-training'):

    assert mode in ['plotting', 'damaging-training']

    ### loading wac
    if args.language == 'en':
        identifier = 'PukWaC'
    elif args.language == 'it':
        identifier = 'itwac'
    elif args.language == 'de':
        identifier = 'sdewac-v3-tagged'

    ### adding detail for relu
    if 'relu' in args.function:
        function_marker = '{}{}'.format(args.function, args.relu_base)
    else:
        function_marker = args.function
    setup_info = '{}_{}_{}'.format(
                 args.semantic_modality, 
                 function_marker,
                 args.sampling,
                 )
    if mode == 'plotting':
        file_names = []
    else:
        wac_folder = os.path.join(
                                  args.corpora_path, 
                                  args.language, 
                                  '{}_smaller_files'.format(identifier)
                                  )
        assert os.path.exists(wac_folder)
        out_wac = os.path.join(
                               args.corpora_path, 
                               args.language, 
                               'damaged', 
                               '{}_smaller_files_damaged_{}'.format(identifier, setup_info),
                                           )
        os.makedirs(out_wac, exist_ok=True)
        ### loading opensubs
        opensubs_folder = os.path.join(
                                       args.corpora_path, 
                                       args.language, 
                                       'opensubs_ready',
                                       )
        assert os.path.exists(opensubs_folder)
        out_opensubs = os.path.join(
                                    args.corpora_path, 
                                    args.language, 
                                    'damaged', 
                                    'opensubs_ready_damaged_{}'.format(setup_info),
                                    )
        os.makedirs(out_opensubs, exist_ok=True)

        file_names = [[os.path.join(
                                    wac_folder, 
                                    f
                                    ), 
                       os.path.join(
                                    out_wac, 
                                    f
                                    )] for f in os.listdir(wac_folder)] + [[os.path.join(
                                                 opensubs_folder, 
                                                 f
                                                 ), os.path.join(
                                                 out_opensubs, 
                                                 f
                                               )] for f in os.listdir(opensubs_folder)] 
    return file_names, setup_info

def read_args(mode):
    assert mode in ['damage', 'prediction', 'results', 'training']
    parser = argparse.ArgumentParser()
    parser.add_argument('--debugging', 
                        action='store_true',
                        )
    parser.add_argument('--language', 
                        choices=[
                                 'it', 
                                 'en', 
                                 'de',
                                 ], 
                        required=True
                        )
    parser.add_argument('--model', 
                        choices=[
                                 'fasttext', 
                                 'w2v', 
                                 ], 
                        default='w2v',
                        )
    if mode != 'results':
        parser.add_argument(
                            '--corpora_path', 
                            required=True,
                            help='path to the folder containing '
                            'the files for all the languages/corpora'
                            )
    if mode in ['damage', 'training']:
        parser.add_argument('--semantic_modality', 
                            choices=[
                                     'auditory', 
                                     'action',
                                     ], 
                            required=True,
                            )
        parser.add_argument('--relu_base', 
                            choices=[
                                     '50', 
                                     '75', 
                                     '90',
                                     '95',
                                     ], 
                            default='90',
                            )
        parser.add_argument('--sampling', 
                            choices=[
                                     'random', 
                                     'inverse', 
                                     'pos',
                                     ], 
                            default='random',
                            )
        parser.add_argument('--function', 
                            choices=[
                                     'sigmoid', 
                                     'raw', 
                                     'exponential', 
                                     'relu-raw', 
                                     'relu-raw-thresholded80', 
                                     'relu-raw-thresholded85', 
                                     'relu-raw-thresholded90', 
                                     'relu-raw-thresholded95', 
                                     'relu-raw-thresholded99', 
                                     'relu-exponential', 
                                     'logarithmic', 
                                     'relu-logarithmic', 
                                     'relu-sigmoid', 
                                     'relu-step',
                                     ], 
                            required=True,
                            )
    args = parser.parse_args()
    return args
