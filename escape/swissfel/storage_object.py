class StorageObject:
    def __init__(self, factory, *args, **kwargs):
        self.factory = factory
        self.args = args
        self.kwargs = kwargs


class NamespaceCitizen:
    def __init__(self, namespace_object):
        if not hasattr(namespace_object, "alias"):
            raise Exception("Namspace citizen requires a name defined by Alias")

        self.namespace_object = namespace_object

    @property
    def name(self):
        return self.namespace_object.alias.get_full_name()


import json, pickle


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, StorageObject):
            d = {"type": "StorageObject"}
            d["factory"] = pickle.dumps(StorageObject.factory)
            d["args"] = StorageObject.args
            d["kwargs"] = StorageObject.kwargs
            return d
        elif isinstance(obj, NamespaceCitizen):
            d = {"type": "NamespaceCitizen"}
            d["name"] = NamespaceCitizen.names
            return d
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
