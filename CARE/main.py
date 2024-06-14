
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

from typing_extensions import Annotated

from sciutil import SciUtil

u = SciUtil()
app = typer.Typer()


@app.command()
def task2(pretrained_dir: Annotated[str, typer.Argument(help="Where the pretrained data is located.")], 
          output_dir: Annotated[str, typer.Argument(help="Where you want results to be saved.")], 
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
    
    query_datasets = ["easy", "medium", "hard"]
    if query_dataset not in query_datasets:
        raise typer.BadParameter(f"Invalid query_dataset. Choose from {query_datasets}.")
        
    tasker = Task2(f'{pretrained_dir}splits/task2/', output_dir, f'{pretrained_dir}processed_data/', pretrained_dir=f'{pretrained_dir}task2_baselines/')
    tasker.downstream_retrieval(baseline, reference_dataset, query_dataset, reference_modality, query_modality)
    # For now print the DF
    print(tasker.tabulate_results(baseline, query_dataset))
    u.dp(["Done benchmark. Results are saved in: ", output_dir])


@app.command()
def task1(pretrained_dir: Annotated[str, typer.Argument(help="Where the pretrained data is located.")], 
          output_dir: Annotated[str, typer.Argument(help="Where you want results to be saved.")], 
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
    baselines =["BLAST", "CLEAN", "ChatGPT", "ProteInfer"]
    if baseline not in baselines:
        raise typer.BadParameter(f"Invalid baseline. Choose from {baselines}.")
    
    query_datasets = ["30", "30-50", "price", "promiscuous"]
    if query_dataset not in query_datasets:
        raise typer.BadParameter(f"Invalid query_dataset. Choose from {query_datasets}.")
        
    tasker = Task1(data_folder=f'{pretrained_dir}splits/task1/', output_folder=output_dir, processed_data_folder=f'{pretrained_dir}processed_data/')
    # For proteInfer you need to point where it was saved.
    df = tasker.get_blast(query_dataset, num_ecs=k, save=True)

    # For now print the DF (maybe for debugging)
    print(df.head())

    u.dp(["Done benchmark. Results are saved in: ", output_dir])

if __name__ == "__main__":
    app()