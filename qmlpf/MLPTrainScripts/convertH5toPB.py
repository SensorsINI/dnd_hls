## This file is part of https://github.com/SensorsINI/dnd_hls.
## This intellectual property is licensed under the terms of the project license available at the root of the project.

# use tf2.x(tf1.14 also ok) to convert the model hdf5 trained by tf2.x to pb
import tensorflow as tf
import os,sys
def freeze_session(model_path=None,clear_devices=True):
    print(f'converting h5 model to pb model for model_path={model_path}')
    tf.compat.v1.reset_default_graph()
    session=tf.compat.v1.keras.backend.get_session()
    graph = session.graph
    with graph.as_default():
        model = tf.keras.models.load_model(model_path)
        output_names = [out.op.name for out in model.outputs]
        print("output_names",output_names)
        input_names =[innode.op.name for innode in model.inputs]
        print("input_names",input_names)
        input_graph_def = graph.as_graph_def()
        # for node in input_graph_def.node:
        #     print('node:', node.name)
        print("len node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)#去掉与推理无关的内容
        # print("##################################################################")
        # for node in outgraph.node:
        #     print('node:', node.name)
        # print("length of  node",len(outgraph.node))
        (filepath,filename) = os.path.split(model_path)
        # logdir= "./2xpb/"
        from pathlib import Path
        p=Path(model_path)
        logdir=str(p.parent)
        pbfilename=filename.replace('.h5', '.pb')
        print(f'saving model to {logdir}/{pbfilename}')
        tf.io.write_graph(frozen_graph, logdir, pbfilename, as_text=False)
        print(f'*** wrote .pb model for jAER to {logdir}/{pbfilename}')
        return outgraph

def main(h5_path):  
    if not os.path.isdir('./2xpb/'):
        os.mkdir('./2xpb/')
    freeze_session(h5_path,True)

if __name__ == "__main__":
    if len(sys.argv)<2:
        from easygui import fileopenbox
        f=fileopenbox('select h5 model file', default='models/*.h5',title='h5 chooser')
    else:
        f=sys.argv[1]
    main(f)
