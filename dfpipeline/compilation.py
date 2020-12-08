##############################################################################
# Copyright 2020 IBM Corp. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
##############################################################################

import networkx as nx
from pprint import pprint

class nodes(nx.DiGraph):
    def getChildIndices(self, node):
       return [p for p in self.predecessors(node)]

    def getParentIndices(self, node):
       return [p for p in self.successors(node)]

    def append(self, **kwargs):
        ret = nx.number_of_nodes(self)
        self.add_node(ret, **kwargs)
        return ret

    def showNode(self, i, depth):
        node = self.nodes[i]
        '''
        phi = node.get('phi')
        phi = 'phi='+phi if phi else ''
        '''
        arg = 'arg='+(str(node['arg']) if type(node['arg']) != object else '')
        indentStr = ':{0:' + str(depth*3+1) + '}'
        print('{0:>6}'.format(i), indentStr.format(''), node['name'], arg, "children=", self.getChildIndices(i), "parents=", self.getParentIndices(i))


class block():
    def __init__(self, compilation, bbNum):
        super().__init__()
        self.compilation = compilation
        self.bbNum = bbNum
        self.insts = []
        self.inst_depths = {}

    def getBBNum(self):
        return self.bbNum

    def append(self, inst):
        self.insts.append(inst)

    def insertBefore(self, tgt, inst):
        idx = self.insts.index(tgt)
        self.insts.insert(idx, inst)

    def insertAfter(self, tgt, inst):
        idx = self.insts.index(tgt)
        self.insts.insert(idx+1, inst)

    def getInsts(self):
        return self.insts

    def inBlock(self, pos):
        return pos in self.insts

    def pred(self):
        return self.compilation.cfg().pred(self.getBBNum())

    def succ(self):
        return self.compilation.cfg().succ(self.getBBNum())

    def computeInstDepthSub(self, i, depth):
        for c in self.compilation.getChildIndices(i):
            self.computeInstDepthSub(c, depth+1)
        self.inst_depths[i] = depth

    def computeInstDepth(self):
        self.inst_depths = {}
        for i in self.insts:
            if len(self.compilation.getParentIndices(i)) == 0:
                self.computeInstDepthSub(i, 0)

    def dumpAllInsts(self):
        print(self)
        self.computeInstDepth()
        for i in self.insts:
            depth = self.inst_depths[i]
            self.compilation.nodes().showNode(i, depth)

    def __str__(self):
        return "BB" + str(self.bbNum) + " pred=" + str(self.pred()) + " succ=" + str(self.succ())


class BlockInfo(object):
    def __init__(self, block, bbnum, incoming_blocks, nodes):
        self.block = block
        self.bbnum = bbnum
        # The list of incoming BlockInfo objects (obtained by control
        # flow analysis).
        self.incoming_blocks = incoming_blocks
        self.stack = []
        # Outgoing variables from this block:
        #   { outgoing phi name -> var name }
        self.outgoing_phis = {}
        self.insts = []
        self.tempct = 0
        self._term = None
        self.stack_offset = None
        self.stack_effect = 0
        self.syntax_blocks = None
        self.bcindex = None
        self.nodes = nodes

    def __repr__(self):
        return "<%s at bbnum %d>" % (self.__class__.__name__, self.bbnum)

    def dump(self):
        print("bbnum", self.bbnum, "{")
        print("  stack: ", end='')
        pprint(self.stack)
        pprint(self.insts)
        pprint(self.outgoing_phis)
        print("}")

    def make_temp(self, prefix=''):
        self.tempct += 1
        name = '_%s%s_%s_%s' % (prefix, self.bbnum, self.tempct, self.bcindex)
        return name

    def getBCidx(cls, temp_string):
        return int(temp_string.split('_')[-1])

    def push(self, val):
        self.stack_effect += 1
        self.stack.append(val)

    def pop(self, discard=False):
        """
        Pop a variable from the stack, or request it from incoming blocks if
        the stack is empty.
        If *discard* is true, the variable isn't meant to be used anymore,
        which allows reducing the number of temporaries created.
        """
        if not self.stack:
            self.stack_offset -= 1
            if not discard:
                ret = self.make_incoming()
                new_node = self.nodes.append(name='LOAD_FAST', arg=ret)
                self.nodes.add_edge(new_node, self.bcindex)
                return ret
        else:
            self.stack_effect -= 1
            ret = self.stack.pop()
            fromEdge = self.getBCidx(ret)
            self.nodes.add_edge(fromEdge, self.bcindex)
            return ret

    def peek(self, k):
        """
        Return the k'th element back from the top of the stack.
        peek(1) is the top of the stack.
        """
        num_pops = k
        top_k = [self.pop() for _ in range(num_pops)]
        r = top_k[-1]
        for i in range(num_pops - 1, -1, -1):
            self.push(top_k[i])
        return r

    def make_incoming(self):
        """
        Create an incoming variable (due to not enough values being
        available on our stack) and request its assignment from our
        incoming blocks' own stacks.
        """
        assert self.incoming_blocks
        ret = self.make_temp('phi')
        for ib in self.incoming_blocks:
            stack_index = self.stack_offset + self.stack_effect
            ib.request_outgoing(self, ret, stack_index)
        return ret

    def request_outgoing(self, outgoing_block, phiname, stack_index):
        """
        Request the assignment of the next available stack variable
        for block *outgoing_block* with target name *phiname*.
        """
        if phiname in self.outgoing_phis:
            # If phiname was already requested, ignore this new request
            # (can happen with a diamond-shaped block flow structure).
            return
        if stack_index < self.stack_offset:
            assert self.incoming_blocks
            for ib in self.incoming_blocks:
                ib.request_outgoing(self, phiname, stack_index)
        else:
            varname = self.stack[stack_index - self.stack_offset]
            index = self.getBCidx(varname)
            new_node = self.nodes.append(name='STORE_FAST', arg=phiname)
            self.nodes.add_edge(index, new_node)
            self.outgoing_phis[phiname] = varname

    @property
    def tos(self):
        r = self.pop()
        self.push(r)
        return r

    def append(self, inst, **kws):
        self.insts.append((self.bcindex, inst, kws))

    @property
    def terminator(self):
        assert self._term is None
        return self._term

    @terminator.setter
    def terminator(self, inst):
        self._term = inst


class CFG(nx.DiGraph):
    def __init__(self, compilation):
        super().__init__()
        self.compilation = compilation
        self.__bbcnt = 0
        self.entryBlock = self.addBB(-1, -1)
        self.exitBlock = None

    def makeExitBlock(self):
        if self.__bbcnt >= 2:
            self.add_edge(self.entryBlock.getBBNum(), 1)
        self.exitBlock = self.addBB(-1, -1)
        for i in self.nodes:
            if i is not self.exitBlock.getBBNum() and not any(True for _ in self.successors(i)):
                self.add_edge(i, self.exitBlock.getBBNum())

    def setEntry(self, e):
        self.entryBlock = e

    def setExit(self, e):
        self.exitBlock = e

    def getBBCnt(self):
        return self.__bbcnt

    def incBBCnt(self):
        org = self.__bbcnt
        self.__bbcnt += 1
        return org

    def addBB(self, start, end):
        bb = block(self.compilation, self.incBBCnt())
        if start >= 0:
            bb.getInsts().extend([x for x in range(start, end+1)])
        self.add_node(bb.getBBNum(), bb = bb)
        return bb

    def findBlock(self, nodepos):
        for i in self.nodes:
            bb = self.nodes[i]['bb']
            if bb.inBlock(nodepos):
                return bb
        return None

    def findBBNum(self, nodepos):
        ret = self.findBlock(nodepos)
        if ret is not None:
            return ret.getBBNum()
        return None

    def pred(self, b):
        return [p for p in self.predecessors(b)]

    def succ(self, b):
        return [s for s in self.successors(b)]

    def __str__(self):
        ret = 'CFG\n'
        for i in self.nodes:
            ret += "BB" + str(i) + " pred=" + str(self.pred(i)) + " succ=" + str(self.succ(i)) + "\n"
        return ret


class compilation:
    def __init__(self, bytecode):
        self.__originalBytecode = bytecode
        self.__nodes = nodes()
        self.__CFG = CFG(self)

    def bytecode(self):
        return self.__originalBytecode

    def nodes(self):
        return self.__nodes

    def cfg(self):
        return self.__CFG

    def dumpCFGandInst(self):
        cfg = self.cfg()
        for i in cfg.nodes:
            bb = cfg.nodes[i]['bb']
            bb.dumpAllInsts()

    # Useful short cuts
    def getBlock(self, bi): return self.cfg().nodes[bi]['bb']
    def getPredIndices(self, bi): return self.cfg().pred(bi)
    def getSuccIndices(self, bi): return self.cfg().succ(bi)

    def getNode(self, ni): return self.nodes().nodes[ni]
    def getChildIndices(self, ni): return self.nodes().getChildIndices(ni)
    def getParentIndices(self, ni): return self.nodes().getParentIndices(ni)
    def getChildIndex(self, ni, j): return self.getChildIndices(ni)[j]
    def getParentIndex(self, ni, j): return self.getParentIndices(ni)[j]
