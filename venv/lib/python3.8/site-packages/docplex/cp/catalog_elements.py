# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the different object classes used to describe the
CPO catalog of types and functions.
"""

###############################################################################
## Public classes
###############################################################################

class CpoType(object):
    """ CPO type (flavor) descriptor """
    __slots__ = ('name',               # Name of the type
                 'is_variable',        # Indicate a type corresponding to a variable
                 'is_constant',        # Indicates that type denotes a constant (possibly array)
                 'is_constant_atom',   # Indicates that type denotes an atomic constant
                 'is_array',           # Indicates that type describes an array
                 'is_array_of_expr',   # Indicates that type describes an array of expressions (not constants)
                 'is_toplevel',        # Indicates that type describes an expression that can be used as top level expression
                 'higher_types',       # List of higher types in the hierarchy
                 'element_type',       # Type of array element (for arrays)
                 'parent_array_type',  # Type corresponding to an array of this type
                 'base_type',          # Base type to be used for signature matching
                 'id',                 # Unique type id (index) used to fasten type links
                 'kind_of_types',      # Set of types that are kind of this one
                 'common_types'        # Dictionary of types that are common with this and others.yield
                                       # Key is type name, value is common type with this one.
                )

    def __init__(self, name, isvar=False, iscst=False, istop=False, isatm=False, htyps=(), eltyp=None, bastyp=None):
        """ Create a new type definition

        Args:
            name:  Name of the type
            isvar: Indicates whether this type denotes a variable
            htyps: List of types higher in the hierarchy
            eltyp: Array element type, None (default) if not array
            iscst: Indicate whether this type denotes a constant
            isatm: Indicate whether this type denotes an atomic constant
        """
        super(CpoType, self).__init__()
        self.name              = name
        self.is_variable       = isvar
        self.is_constant       = iscst
        self.is_constant_atom  = isatm
        self.is_toplevel       = istop
        self.higher_types = (self,) + htyps
        self.element_type = eltyp
        if bastyp is None:
            self.base_type = self
        else:
            self.base_type = bastyp
            self.is_variable = bastyp.is_variable
            self.is_constant = bastyp.is_constant
            self.is_constant_atom = bastyp.is_constant_atom
        # Process array case
        if eltyp is not None:
            eltyp.parent_array_type = self
            self.is_constant  = eltyp.is_constant
        self.is_array = eltyp is not None
        self.is_array_of_expr = self.is_array and not self.is_constant
        self.parent_array_type = None

    def get_name(self):
        """ Get the name of the type

        Returns:
            Name of the type
        """
        return self.name

    def is_kind_of(self, tp):
        """ Check if this type is a kind of another type, i.e. other type is in is hierarchy

        Args:
            tp: Other type to check
        Returns:
            True if this type is a kind of tp
        """
        # Check if required type is the same
        # return tp.base_type in self.higher_types
        return self.kind_of_types[tp.id]

    def get_common_type(self, tp):
        """ Get first common type between this and the parameter.
        
        Args:
            tp: Other type 
        Returns:
            The first common type between this and the parameter, None if none
        """
        return self.common_types[tp.id]

    def _compute_common_type(self, tp):
        """ Compute the first common type between this and the parameter.

        Args:
            tp: Other type
        Returns:
            The first common type between this and the parameter, None if none
        """
        # Check if given type is derived
        if tp.base_type is not tp:
            tp = tp.base_type
        # Check if this type is derived
        if self.base_type is not self:
            return self.base_type._compute_common_type(tp)
        # Check direct comparison
        if (self is tp) or tp.is_kind_of(self):
            return self
        elif self.is_kind_of(tp):
            return tp
        # Search common types in ancestors
        for ct in self.higher_types:
            if ct in tp.higher_types:
                return ct
        return None

    def __str__(self):
        """ Convert this object into a string """
        return self.name

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.name == other.name))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)

    def __hash__(self):
        """ Return object hash-code """
        return id(self)


class CpoParam(object):
    """ Descriptor of an operation parameter """
    __slots__ = ('type',          # Parameter CPO type
                 'default_value'  # Parameter default value, None if none
                )
    
    def __init__(self, ptyp, dval=None):
        """ Create a new parameter
        
        Args:
            ptyp: Parameter type
            dval: Default value
        """
        super(CpoParam, self).__init__()
        self.type = ptyp
        self.default_value = dval
        
    def __str__(self):
        if self.default_value is None:
            return self.type.name
        else:
            return self.type.name + "=" + str(self.default_value)

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.type == other.type) and (self.default_value == other.defval))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


# Marker to signal any type and number of arguments
TYPE_ANY = CpoType("Any")
ANY_ARGUMENTS = (CpoParam(TYPE_ANY),)

class CpoSignature(object):
    """ Descriptor of a signature of a CPO operation """
    __slots__ = ('return_type',  # Return type
                 'parameters',   # List of parameter descriptors
                 'operation'     # Parent operation
                 )
    
    def __init__(self, rtyp, ptyps):
        """ Create a new signature
        
        Args:
            rtyp:  Returned type
            ptyps: Array of parameter types
        """
        super(CpoSignature, self).__init__()
        self.return_type = rtyp
        
        # Build list of parameters
        if ptyps is ANY_ARGUMENTS:
            self.parameters = ANY_ARGUMENTS
        else:
            lpt = []
            for pt in ptyps:
                if isinstance(pt, CpoParam):
                    lpt.append(pt)
                else:
                    lpt.append(CpoParam(pt))
            self.parameters = tuple(lpt)
        
    def __str__(self):
        return str(self.return_type) + "[" + ", ".join(map(str, self.parameters)) + "]"

    def __eq__(self, other):
        """ Check equality of this object with another """
        if self is other:
            return True
        return isinstance(other, self.__class__) and (self.return_type == other.rtype) \
               and (self.operation == other.operation) and (self.parameters == other.params)

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


class CpoOperation(object):
    """ CPO operation descriptor """
    __slots__ = ('cpo_name',     # Operation CPO name
                 'python_name',  # Operation python name
                 'keyword',      # Operation keyword (operation symbol)
                 'priority',     # Operator priority, -1 for function call
                 'signatures'    # List of possible operation signatures
                 )

    def __init__(self, cpname, pyname, kwrd, prio, signs):
        """ Create a new operation
        
        Args:
            cpname:  Operation CPO name
            pyname:  Operation python name
            kwrd:    Keyword, None for same as cpo name
            prio:    Priority
            signs:   Array of possible signatures
        """
        super(CpoOperation, self).__init__()
        
        # Store attributes
        self.cpo_name = cpname
        self.python_name = pyname
        self.priority = prio
        if kwrd:
            self.keyword = kwrd
        else:
            self.keyword = cpname
        self.signatures = signs
        
        # Set pointer back on operation on each signature
        for s in signs:
            s.operation = self
        
    def __str__(self):
        return str(self.cpo_name) + "(" + ", ".join(map(str, self.signatures)) + ")"

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.cpo_name == other.cpo_name))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


def compute_all_type_links(ltypes):
    """ Compute all links between the different data types.

    Args:
        ltypes: List of all types
    """
    # Allocate id to each each type
    nbtypes = len(ltypes)
    for i, tp in enumerate(ltypes):
        tp.id = i

    # Compute kind of for each type
    for tp1 in ltypes:
        tp1.kind_of_types = tuple(map(lambda tp2: (tp2.base_type in tp1.higher_types), ltypes))

    # Compute common type
    for tp1 in ltypes:
        ctypes = [None] * nbtypes
        for tp2 in ltypes:
            ct = tp1._compute_common_type(tp2)
            if ct is not None:
                ctypes[tp2.id] = ct
        tp1.common_types = tuple(ctypes)

