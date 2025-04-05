## This file is part of https://github.com/SensorsINI/dnd_hls.
## This intellectual property is licensed under the terms of the project license available at the root of the project.

# use tf2.x(tf1.14 also ok) to convert the model hdf5 trained by tf2.x to pb
import tensorflow as tf
import os,sys,re
from weighted_binary_cross_entropy import weighted_binary_crossentropy
from gabor_initializer_1d_patch import gabor_initializer_1d_patch
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2

def remove_dropout_nodes(input_graph):
  """Prunes out nodes that aren't needed for inference.

  There are nodes like Identity and CheckNumerics that are only useful
  during training, and can be removed in graphs that will be used for
  nothing but inference. Here we identify and remove them, returning an
  equivalent graph. To be specific, CheckNumerics nodes are always removed, and
  Identity nodes that aren't involved in control edges are spliced out so that
  their input and outputs are directly connected.

  Args:
    input_graph: Model to analyze and prune.

  Returns:
    A list of nodes with the unnecessary ones removed.
  """
  types_to_remove = {"CheckNumerics": True}

  input_nodes = input_graph.node
  names_to_remove = {}
  for node in input_nodes:
    if 'dropout' in node.name:
      names_to_remove[node.name] = True

  nodes_after_removal = []
  for node in input_nodes:
    if node.name in names_to_remove:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      if input_name in names_to_remove:
        continue
      new_node.input.append(full_input_name)
    nodes_after_removal.append(new_node)

  types_to_splice = {"Identity": True}
  control_input_names = set()
  node_names_with_control_input = set()
  node_in_colocated = set()

  for node in nodes_after_removal:
    for node_input in node.input:
      if "^" in node_input:
        control_input_names.add(node_input.replace("^", ""))
        node_names_with_control_input.add(node.name)
    # Prevent colocated nodes from being lost.
    if "_class" in node.attr:
      for colocated_node_name in node.attr["_class"].list.s:
        node_in_colocated.add(_get_colocated_node_name(colocated_node_name))

  names_to_splice = {}
  for node in nodes_after_removal:
    if node.op in types_to_splice:
      if node.name in node_in_colocated:
        continue
      # We don't want to remove nodes that have control edge inputs, because
      # they might be involved in subtle dependency issues that removing them
      # will jeopardize.
      if node.name not in node_names_with_control_input:
        names_to_splice[node.name] = node.input[0]

  # We also don't want to remove nodes which are used as control edge inputs.
  names_to_splice = {name: value for name, value in names_to_splice.items()
                     if name not in control_input_names}

  nodes_after_splicing = []
  for node in nodes_after_removal:
    if node.name in names_to_splice:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      while input_name in names_to_splice:
        full_input_name = names_to_splice[input_name]
        input_name = re.sub(r"^\^", "", full_input_name)
      new_node.input.append(full_input_name)
    nodes_after_splicing.append(new_node)

  output_graph = graph_pb2.GraphDef()
  output_graph.node.extend(nodes_after_splicing)
  return output_graph

def _get_colocated_node_name(colocated_node_name):
  """Decodes colocated node name and returns it without loc:@ prepended."""
  colocated_node_decoded = colocated_node_name.decode("utf-8")
  if colocated_node_decoded.startswith("loc:@"):
    return colocated_node_decoded[5:]
  return colocated_node_decoded

def freeze_session(model_path=None,clear_devices=True, model=None):
    print(f'converting h5 model to pb model for model_path={model_path} or model={model}')
    tf.compat.v1.reset_default_graph()
    # from tensorflow.compat.v1.keras import backend as K
    # session=K.get_session()
    session=tf.compat.v1.keras.backend.get_session() # tf.compat.v1.keras.backend.get_session()]
    graph = session.graph
    with graph.as_default():
        if not model_path is None and model is None:
          custom_objects={"_weighted_binary_crossentropy": weighted_binary_crossentropy(), "_gabor_initializer_1d_patch": gabor_initializer_1d_patch()}
          model = tf.saved_model.load(model_path)
          # model = tf.compat.v1.keras.models.load_model(model_path, custom_objects=custom_objects)
        # https://stackoverflow.com/questions/4260280/if-else-in-a-list-comprehension
        output_names = [v.name[:-2] for v in model.variables.variables if ('output/bias' in v.name) ]
        print("output_names",output_names)
        # input_names =[innode.op.name for innode in model.inputs]
        # print("input_names",input_names)
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            print('         node:', node.name)
        print("length of node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        # frozen_graph =  tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
        #                                               output_names)
        
        # outgraph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)#去掉与推理无关的内容
        # # outgraph=remove_dropout_nodes(outgraph)
        # # print("##################################################################")
        # for node in outgraph.node:
        #     print('node:', node.name)
        # print("length of  node",len(outgraph.node))
        # (filepath,filename) = os.path.split(model_path)
        # logdir= "./2xpb/"
        from pathlib import Path
        p=Path(model_path)
        logdir=str(p.parent)
        # pbfilename=filename.replace('.h5', '.pb')
        pbfilename=model_path+'.pb'
        print(f'saving model to {pbfilename}')
        # tf.io.write_graph(frozen_graph, logdir, pbfilename, as_text=False)
        tf.io.write_graph(input_graph_def, logdir, pbfilename, as_text=False)
        print(f'*** wrote .pb model for jAER to {logdir}/{pbfilename}')
        # return outgraph
        return input_graph_def

def main(h5_path):  
    if not os.path.isdir('./2xpb/'):
        os.mkdir('./2xpb/')
    freeze_session(h5_path,True)

if __name__ == "__main__":
    from prefs import MyPreferences
    prefs=MyPreferences()
    if len(sys.argv)<2:
        from easygui import fileopenbox,diropenbox
        # f=fileopenbox('select h5 model file', default='models/*.h5',title='h5 chooser')
        f=diropenbox('select exported model folder', default=prefs.get("lastdir",'models/'),title='model dir chooser', )
        if f is None or f=='':
            # print('no file selected')
            print('no dir selected')
            quit(0)
        prefs.put('lastdir',f)
    else:
        f=sys.argv[1]
    main(f)
