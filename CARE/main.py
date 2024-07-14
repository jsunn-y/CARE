
#################################################################################
# Copyright (c) 2012-2024 Scott Chacon and others                               #
#                                                                               #    
# Permission is hereby granted, free of charge, to any person obtaining         #
# a copy of this software and associated documentation files (the               #
# "Software"), to deal in the Software without restriction, including           #
# without limitation the rights to use, copy, modify, merge, publish,           #
# distribute, sublicense, and/or sell copies of the Software, and to            #
# permit persons to whom the Software is furnished to do so, subject to         #
# the following conditions:                                                     #                                          
# The above copyright notice and this permission notice shall be                #
# included in all copies or substantial portions of the Software.               #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,               #
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF            #
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                         #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE        #  
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION        #
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION         #
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.               #
#################################################################################


import typer

from CARE.processing import *
from CARE.task2 import *
from CARE.task1 import *
from CARE.benchmarker import *

from typing_extensions import Annotated

from sciutil import SciUtil

u = SciUtil()
app = typer.Typer()


@app.command()
def task2(pretrained_dir: str = typer.Option(..., help="Where the pretrained data is located", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="pretrained dir", 
                                 metavar="pretrained"), 
          output_dir: str = typer.Option(..., help="Where you want your results.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Where you want your results", 
                                 metavar="output"), 
          baseline: str = typer.Option(..., help="The baseline to run.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Choose a baseline", 
                                 metavar="baseline"),
         query_dataset: str = typer.Option(..., help="The query dataset to run.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Choose a query_dataset", 
                                 metavar="query_dataset"), 
                                 #choices=["easy", "medium", "hard"]),
         reference_dataset: str = typer.Option(..., help="The reference dataset to query against.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Choose a reference_dataset", 
                                 metavar="reference_dataset"), 
                                 #choices=["all_ECs"]),
         query_modality: str = typer.Option(..., help="The query modaliy used.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Choose a query_modality", 
                                 metavar="query_modality"), 
                                 #choices=["reaction", "protein", "text"]),
         reference_modality: str = typer.Option(..., help="The reference_modality in the dataset.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Choose a reference_modality", 
                                 metavar="reference_modality")):
    
    """ Run Task 2 """
    u.dp(["Running task 2... ", baseline, query_dataset])

    baselines =["Similarity", "CARE", "CLIPZyme", "CREEP"]
    if baseline not in baselines:
        raise typer.BadParameter(f"Invalid baseline. Choose from {baselines}.")
    
    query_modalitys = ["reaction", "protein", "text"]
    if query_modality not in query_modalitys:
        raise typer.BadParameter(f"Invalid query_modality. Choose from {query_modalitys}.")
    
    reference_modalitys = ["reaction", "protein", "text"]
    if reference_modality not in reference_modalitys:
        raise typer.BadParameter(f"Invalid reference_modality. Choose from {reference_modalitys}.")
    
    if reference_dataset not in ['all_ECs']:
        raise typer.BadParameter(f"Invalid reference_dataset. Choose from all_ECS.")
    
    query_datasets = ["easy", "medium", "hard", 'All']
    if query_dataset not in query_datasets:
        raise typer.BadParameter(f"Invalid query_dataset. Choose from {query_datasets}.")
        
    if baseline == 'All':
        for baseline in baselines:
            if query_dataset == 'All':
                for query_dataset in query_datasets:
                    u.dp(['Running: ', baseline, query_dataset])
                    tasker = Task2(f'{pretrained_dir}splits/task2/', output_dir, f'{pretrained_dir}processed_data/', pretrained_dir=f'{pretrained_dir}task2_baselines/')
                    tasker.downstream_retrieval(baseline, reference_dataset, query_dataset, reference_modality, query_modality)
                    u.warn_p(['Done: ', baseline, query_dataset])
                    tasker.tabulate_results(baseline, query_dataset)

            else:
                u.dp(['Running: ', baseline, query_dataset])
                tasker = Task2(f'{pretrained_dir}splits/task2/', output_dir, f'{pretrained_dir}processed_data/', pretrained_dir=f'{pretrained_dir}task2_baselines/')
                tasker.downstream_retrieval(baseline, reference_dataset, query_dataset, reference_modality, query_modality)
                u.warn_p(['Done: ', baseline, query_dataset])
                tasker.tabulate_results(baseline, query_dataset)


    elif query_dataset == 'All':
        for query_dataset in query_datasets:
            u.dp(['Running: ', baseline, query_dataset])
            tasker = Task2(f'{pretrained_dir}splits/task2/', output_dir, f'{pretrained_dir}processed_data/', pretrained_dir=f'{pretrained_dir}task2_baselines/')
            tasker.downstream_retrieval(baseline, reference_dataset, query_dataset, reference_modality, query_modality)
            tasker.tabulate_results(baseline, query_dataset)

            u.warn_p(['Done: ', baseline, query_dataset])
    else:
        # Just run the single task!
        tasker = Task2(f'{pretrained_dir}splits/task2/', output_dir, f'{pretrained_dir}processed_data/', pretrained_dir=f'{pretrained_dir}task2_baselines/')
        tasker.downstream_retrieval(baseline, reference_dataset, query_dataset, reference_modality, query_modality)
        df = tasker.tabulate_results(baseline, query_dataset)
        rows = []
        rows = get_k_acc(df, [1, 5, 10], rows, baseline, query_dataset)
        results_df = pd.DataFrame(rows, columns=['baseline', 'split', 'k', 'level 4 accuracy', 'level 3 accuracy', 'level 2 accuracy', 'level 1 accuracy'])    
        results_df.to_csv(f'{output_dir}results_all.csv')
        u.dp(["Done benchmark. Results are saved in: ", output_dir])


def run_task_1(tasker, baseline, split, rows):
    if baseline == 'BLAST':
        # For proteInfer you need to point where it was saved.
        df = tasker.get_blast(split, num_ecs=10, save=True)
        u.dp([split])
        rows = get_k_acc(df, [1, 5, 10], rows, 'BLAST', split)
    elif baseline == 'ProteInfer':
        # For proteInfer you need to point where it was saved.
        df = tasker.get_proteinfer(split, proteinfer_dir=f'{tasker.data_folder}task1_baselines/ProteInfer/proteinfer/', save=True)
        rows = get_k_acc(df, [1, 5, 10], rows, 'ProteInfer', split)
    elif baseline == 'Random':
        df = tasker.randomly_assign_EC(split, num_ecs=50, save=True)
        rows = get_k_acc(df, [1, 5, 10], rows, 'random', split)
    elif baseline == 'ChatGPT':
        df = tasker.get_ChatGPT(split, num_ecs=50, save=True)
        rows = get_k_acc(df, [1, 5, 10], rows, 'ChatGPT', split)
    elif baseline == 'CLEAN':
        u.dp(["For CLEAN please follow the instructions in the notebook."])
    return rows

@app.command()
def task1(pretrained_dir: str = typer.Option(..., help="Where the pretrained data is located", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="pretrained dir", 
                                 metavar="pretrained"), 
          output_dir: str = typer.Option(..., help="Where you want your results.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Where you want your results", 
                                 metavar="output"), 
          baseline: str = typer.Option(..., help="The baseline to run.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Choose a baseline", 
                                 metavar="baseline"),
         query_dataset: str = typer.Option(..., help="The query dataset to run.", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Choose a query_dataset", 
                                 metavar="query_dataset"), 
         k: int = typer.Option(..., help="Number of ECs to get (1-N)", 
                                 case_sensitive=False, 
                                 show_choices=True, 
                                 prompt="Choose a number of ranked ECS", 
                                 metavar="k")):
    
    """ Run Task 1 """
    rows = []

    baselines =["BLAST", "CLEAN", "ChatGPT", "ProteInfer", "Random"]
    if baseline not in baselines:
        raise typer.BadParameter(f"Invalid baseline. Choose from {baselines}.")
    
    query_datasets = ["30", "30-50", "price", "promiscuous", "Random"]
    if query_dataset not in query_datasets:
        raise typer.BadParameter(f"Invalid query_dataset. Choose from {query_datasets}.")
        
    if baseline == 'All':
        for baseline in baselines:
            if query_dataset == 'All':
                for query_dataset in query_datasets:
                    u.dp(['Running: ', baseline, query_dataset])
                    tasker = Task1(data_folder=f'{pretrained_dir}splits/task1/', output_folder=output_dir, processed_data_folder=f'{pretrained_dir}processed_data/')
                    # For proteInfer you need to point where it was saved.
                    rows = run_task_1(tasker, baseline, query_dataset, rows)
                    u.warn_p(['Done: ', baseline, query_dataset])
            else:
                tasker = Task1(data_folder=f'{pretrained_dir}splits/task1/', output_folder=output_dir, processed_data_folder=f'{pretrained_dir}processed_data/')
                # For proteInfer you need to point where it was saved.
                rows = run_task_1(tasker, baseline, query_dataset, rows)

    elif query_dataset == 'All':
        for query_dataset in query_datasets:
            u.dp(['Running: ', baseline, query_dataset])
            tasker = Task1(data_folder=f'{pretrained_dir}splits/task1/', output_folder=output_dir, processed_data_folder=f'{pretrained_dir}processed_data/')
            # For proteInfer you need to point where it was saved.
            rows = run_task_1(tasker, baseline, query_dataset, rows)
            u.warn_p(['Done: ', baseline, query_dataset])
    else:
        # Just run the single task!
        tasker = Task1(data_folder=f'{pretrained_dir}splits/task1/', output_folder=output_dir, processed_data_folder=f'{pretrained_dir}processed_data/')
        rows = run_task_1(tasker, baseline, query_dataset, rows)

    df = pd.DataFrame(rows, columns=['baseline', 'split', 'k', 'level 4 accuracy', 'level 3 accuracy', 'level 2 accuracy', 'level 1 accuracy'])
    df.to_csv(f'{output_dir}results_all.csv')
    u.dp(["Done benchmark. Results are saved in: ", output_dir])

if __name__ == "__main__":
    app()