## Data set
1. Request the complete `had` repositories from Kiyokawa sensei.
2. Install all required components.
3. Copy the `plan_seq_contacts_data.py` file from this backup folder to the cloned `had/ros2_ws/src/sequence_planner` directory.
4. Run the following command:

```bash
conda activate planners && cd /root/ros2_ws/src/sequence_planner/ && export LIBGL_ALWAYS_INDIRECT=1; python3 plan_seq_contacts_data.py case_num
```
### Notes
- The case_num parameter represents the case study. Changing this value will also change the targeted object.
- The original `plan_seq_contacts.py` file from Kiyokawa sensei only generates one sequence. For data set generation, we need multiple sequences. Therefore, this code is an adjusted version of that file.
- You can modify or add sequences in plan_seq_contacts_data.py as needed.
- This backup was created on March 27, 2025. Please note that significant changes may have been made since then, especially in the `contact_planning.py` file.
- The `contact_planning.py` file generates the graph using an edge-based method. We are working on fixing a bug in the vertex-based method to make it an optional feature that can be selected via arguments.


## Surrogate Model
The surrogate Model implementation is based on the work in the [Two-Stage Training of Graph Neural Networks for Graph Classification](https://github.com/manhtuando97/two-stage-gnn/blob/main/README.md) repository.  It is highly recommended to visit the repository for a comprehensive understanding of the model.

### Notes
- The model is implemented in PyTorch, so you will need to follow the installation requirements. The requirements listed here are minimal and may require adjustments. This step was challenging, and my training code may only work with the specific package versions I used. I have listed the packages I used in `requirements.txt`.
- I used a Conda environment named `gnn`.
- The following command is an example of how to run the training:


```bash
conda activate gnn && cd Code && python train.py --epochs=100 --dropout_ratio=0.5 --pooling_ratio=0.5 --num_features=32 --nhid=32 --final_dim=32 --alpha=1.5
```
- I trained many models under different conditions. The `pickle` files are not included in the `dataset` directory due to their large size. These files contain the dataset itself, which you will need to create yourself.