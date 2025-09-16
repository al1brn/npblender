__all__ = ["TreeNode"]

# ====================================================================================================
# Tree Node interface
# ====================================================================================================

class TreeNode:
        
    # ---------------------------------------------------------------------------
    # Unique path
    # ---------------------------------------------------------------------------
    
    @property
    def nd_path(self):
        if self.nd_owner is None:
            return "0"
        else:
            return f"{self.nd_owner.nd_path}.{self.nd_index}"
        
    # ---------------------------------------------------------------------------
    # Iterator
    # ---------------------------------------------------------------------------
    
    def nd_nodes(self):
        yield self
    
        for node in self.nd_child:            
            for n in node:
                yield n
            
    # ---------------------------------------------------------------------------
    # Dynamic properties
    # ---------------------------------------------------------------------------
    
    @property
    def nd_owner(self):
        return getattr(self, '_nd_owner', None)
    
    @nd_owner.setter
    def nd_owner(self, value):
        self._nd_owner = value
        
    @property
    def nd_child(self):
        if not hasattr(self, '_nd_child'):
            self._nd_child = []
        return self._nd_child
            
    # ---------------------------------------------------------------------------
    # Vertical
    # ---------------------------------------------------------------------------
            
    @property
    def nd_is_top(self):
        return self.nd_owner is None
    
    @property
    def nd_depth(self):
        depth = 0
        o = self.nd_owner
        while o is not None:
            depth += 1
            o = o.nd_owner
        return depth
    
    @property
    def nd_top(self):
        o = self
        while o.nd_owner is not None:
            o = o.nd_owner
        return o
    
    # ---------------------------------------------------------------------------
    # Horizontal
    # ---------------------------------------------------------------------------
    
    @property
    def nd_next(self):
        if self.nd_is_top:
            return None
        
        nodes = self.nd_owner.nd_child
        for i, node in enumerate(nodes):
            if node is self:
                return None if i == len(nodes) - 1 else nodes[i+1]
    
    @property
    def nd_prev(self):
        if self.nd_is_top:
            return None
        
        nodes = self.nd_owner.nd_child
        for i, node in enumerate(nodes):
            if node is self:
                return None if i == 0 else nodes[i - 1]
            
    @property
    def nd_index(self):
        if self.nd_is_top:
            return None
        for i, node in enumerate(self.nd_owner.nd_child):
            if node is self:
                return i
        assert(False)
            
    # ---------------------------------------------------------------------------
    # Building
    # ---------------------------------------------------------------------------
            
    def nd_add(self, node):
        node.nd_owner = self
        self.nd_child.append(node)
        return node
    
    def nd_add_after(self, node):
        i = self.nd_index
        if i is None:
            raise AttributeError(f"Impossible to insert a node after the top")

        node.nd_owner = self.nd_owner
        self.nd_owner.nd_child.insert(i + 1, node)
        return node
        
    def nd_add_before(self, node):
        i = self.nd_index
        if i is None:
            raise AttributeError(f"Impossible to insert a node before the top")

        node.owner = self.nd_owner
        self.nd_owner.nd_child.insert(i, node)
        return node
    
    # ---------------------------------------------------------------------------
    # Detacch
    # ---------------------------------------------------------------------------
    
    def nd_detach(self):
        owner = self.nd_owner
        
        if owner is None:
            return self
        
        index = self.nd_index
        if index == 0:
            owner._nd_child = owner._nd_child[1:]
            
        elif index == len(owner._nd_child) - 1:
            owner._nd_child = owner._nd_child[:-1]
            
        else:
            owner._nd_child = owner._nd_child[:index] + owner._nd_child[index + 1:]
            
        self.nd_owner = None
        return self
    