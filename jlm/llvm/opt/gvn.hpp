//
// Created by lars-astrup-sundt on 10/27/25.
//

#ifndef JLM_GVN_H
#define JLM_GVN_H


#ifndef GVN_H
#define GVN_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

#include <stdint.h>



namespace jlm::rvsdg::gvn {
    extern bool gvn_verbose;
    typedef uint64_t GVN_Val;

    constexpr GVN_Val GVN_NULL = 0;
    constexpr GVN_Val GVN_HAS_DEPS                      =  0x1; /* \brief : For making lambda values global */
    constexpr GVN_Val GVN_IS_LOCAL_VALUE                =  0x2;  /* \brief : This value depends on placeholders */
    constexpr GVN_Val GVN_FROM_COLLISION                =  0x4;  /* \brief : This gvn depends on value created by collision resolution */
    constexpr GVN_Val GVN_OP_IS_CA                      =  0x8;  /* \brief : This gvn value represent a commutative and associative operation */
    constexpr GVN_Val GVN_OP_ADDS_PARAM_DEPENDENCE      = 0x10;  /* \brief : This operator creates a local value */
    constexpr GVN_Val GVN_OP_REMOVES_PARAM_DEPENDENCE   = 0x20; /* \brief : For making lambda values global */
    constexpr GVN_Val GVN_OP_IS_SWITCH                  = 0x40;

    constexpr GVN_Val GVN_MASK_INHERIT     = GVN_IS_LOCAL_VALUE | GVN_FROM_COLLISION;
    constexpr GVN_Val GVN_MASK             = GVN_IS_LOCAL_VALUE | GVN_FROM_COLLISION | GVN_OP_IS_CA | GVN_OP_ADDS_PARAM_DEPENDENCE | GVN_OP_REMOVES_PARAM_DEPENDENCE | GVN_HAS_DEPS | GVN_OP_IS_SWITCH;

    inline bool GVN_ValueIsGlobal(GVN_Val v)           {return v & GVN_IS_LOCAL_VALUE;}
    inline bool GVN_ValueIsTainted(GVN_Val v)          {return v & GVN_FROM_COLLISION;}
    inline bool GVN_ValueIsCA_Op(GVN_Val v)            {return v & GVN_OP_IS_CA;}
    inline bool GVN_ValueIsSwitch_Op(GVN_Val v)        {return v & GVN_OP_IS_SWITCH;}
    inline bool GVN_ValueParamDependent(GVN_Val v)     {return v & GVN_OP_ADDS_PARAM_DEPENDENCE;}
    inline bool GVN_ValueIsParamIndependent(GVN_Val v) {return v & GVN_OP_REMOVES_PARAM_DEPENDENCE;}
    inline bool GVN_ValueHasDeps(GVN_Val v)            {return v & GVN_HAS_DEPS;}

    inline std::string to_string(GVN_Val v) {
        auto n = static_cast<uint64_t>(v);
        std::string s = "" + std::to_string(n);
        if ((v & GVN_HAS_DEPS) == 0) {s += "<L>";}
        if (v & GVN_OP_IS_CA) {s+="<ca_operator>";}
        if ((v & GVN_IS_LOCAL_VALUE) == 0) {s += "<global>"; return s;}
        if ((v & GVN_OP_ADDS_PARAM_DEPENDENCE)){s +=    "<op_makes_output_local>";}
        if ((v & GVN_OP_REMOVES_PARAM_DEPENDENCE)){s += "<op_makes_output_global>";}
        if (v & GVN_FROM_COLLISION) {s += "<COLLISION!>"; return s;}
        return s;
    }

    struct BrittlePrismEle {
        GVN_Val partition;
        GVN_Val disruptor;
        GVN_Val original_partition;
        size_t  original_position;
    };

    constexpr const char* ORD_ORIGINAL = "order:original";
    constexpr const char* ORD_PARTITION  = "order:partition";
    constexpr const char* ORD_PARTITION_DISRUPTOR = "order:partition>disruptor";
    constexpr const char* ORD_DISRUPTOR_PARTITION = "order:disruptor>partition";

    class BrittlePrism {
    private:
        const char* current_ordering;

    public:
        bool did_shatter;
        size_t fracture_count;
        std::vector<BrittlePrismEle> elements;

        /// ---------------------------  ORDERINGS -----------------------------------------------------------------
        explicit BrittlePrism(std::vector<GVN_Val> base) : did_shatter(false), fracture_count(0)
        {
            // set gvn values and original indices
            // partition same as unique values
            for (size_t i = 0; i < base.size(); i++) {
                elements.emplace_back(BrittlePrismEle{base[i], base[i], base[i],i});
            }
            current_ordering = ORD_ORIGINAL;
        }

        void OrderByPartition() {
            std::sort(elements.begin(), elements.end(),
                [](BrittlePrismEle& a, BrittlePrismEle& b) {
                    return a.partition < b.partition;
                }
            );
            current_ordering = ORD_PARTITION;
        }

        void OrderByOriginal()
        {
            std::sort(elements.begin(), elements.end(),
                [](BrittlePrismEle& a, BrittlePrismEle& b) {
                    return a.original_position < b.original_position;
                }
            );
            current_ordering = ORD_ORIGINAL;
        }
        void OrderByPartitionThenDisruptor()
        {
            std::sort(elements.begin(), elements.end(),
                [](BrittlePrismEle& a, BrittlePrismEle& b) {
                    if (a.partition == b.partition){return a.disruptor < b.disruptor;}
                    return a.partition < b.partition;
                }
            );
            current_ordering = ORD_PARTITION_DISRUPTOR;
        }
        void OrderByDisruptorThenPartition()
        {
            std::sort(elements.begin(), elements.end(),
            [](BrittlePrismEle& a, BrittlePrismEle& b) {
                    if (a.disruptor == b.disruptor){return a.partition < b.partition;}
                    return a.disruptor < b.disruptor;
                }
            );
            current_ordering = ORD_DISRUPTOR_PARTITION;
        }
        /// ---------------------------  SPANS -----------------------------------------------------------------
        size_t SpanPartition(size_t start)
        {
            size_t span = 1;
            GVN_Val partition = elements[start].partition;
            for (size_t i = start+1; i < elements.size(); i++) {
                if (elements[i].partition == partition) {span++;}else{break;}
            }
            return span;
        }

        size_t SpanDisruptor(size_t start) const
        {
            size_t span = 1;
            GVN_Val disruptor = elements[start].disruptor;
            for (size_t i = start+1; i < elements.size(); i++) {
                if (elements[i].disruptor == disruptor) {span++;}else{break;}
            }
            return span;
        }

        size_t SpanDisruptorAndPartition(size_t start)
        {
            size_t span = 1;
            GVN_Val partition       = elements[start].partition;
            GVN_Val disruptor = elements[start].disruptor;
            for (size_t i = start+1; i < elements.size(); i++) {
                auto e = elements[i];
                if (e.partition == partition && e.disruptor == disruptor) {span++;}else{break;}
            }
            return span;
        }

        /// --------------------------- INSPECT -----------------------------------------------------------------

        void dump() const
        {
            std::cout << "gvn_partition -- " << current_ordering << std::endl;

            for (size_t i = 0; i < elements.size(); i++) {
                auto e = elements[i];
                std::cout << "["<<i<<"] : part("
                    << e.partition <<") pos("
                    << e.original_position << ")  disruptor("
                    << e.disruptor << ")"
                    << std::endl;
            }
        }

        void dump_partitions()
        {
            OrderByPartition();
            size_t it = 0;
            if (elements.empty()){return;}
            while (it < elements.size()) {
                size_t end_it = it + SpanPartition(it);
                while (it < end_it) {
                    std::cout << "(" << elements[it].partition <<":"<< elements[it].original_position << ")";
                    it++;
                }
                std::cout << "  ";
            }
            std::cout << std::endl;
        }
        /// --------------------------- CORE -----------------------------------------------------------------
        void CheckDisruptorInvariants()
        {
            // Partitions should never become more equal
            // No element may become part of another partition at a later stage
            OrderByDisruptorThenPartition();
            size_t it = 0;
            while (it < elements.size()) {
                auto g_span = SpanDisruptorAndPartition(it);
                auto d_span = SpanDisruptor(it);
                if (g_span < d_span) {throw std::runtime_error("Same disruptor value applied into distinct partitions.");}
                it += d_span;
            }
        }

        template<typename Fn>
        static void EachPartition(BrittlePrism& p, Fn cb)
        {
            p.OrderByPartition();
            size_t it = 0;
            while (it < p.elements.size()) {
                size_t p_span = p.SpanPartition(it);
                cb(p.elements[it], p_span);
                it += p_span;
            }
        }

        template<typename Fn>
        static void Shatter(BrittlePrism& p, Fn reassign) {
            if (p.did_shatter){return;}
            // Call this when a loop body has been invoked too many times.
            // This should assign unique partitions to all elements
            // Provable loop invariant variables can be set to GVN_NULL
            p.OrderByOriginal();
            for (size_t i = 0; i < p.elements.size(); i++) {
                reassign(p.elements[i], i);
            }
            p.OrderByPartition();
            for (size_t i = 1; i < p.elements.size(); i++) {
                if (p.elements[i].partition && (p.elements[i].partition == p.elements[i-1].partition) ) {
                    throw std::runtime_error("Shatter failed. All partitions must be unique.");
                }
            }
            p.did_shatter = true;
        }

        template<typename Fn>
        static bool Fracture(BrittlePrism& p, Fn reassign)
        {
            // Reassign take a single (BrittlePrismEle& ele) as argument
            // reassign is called at most once per offending element
            // reassign is NEVER called when the disruptor equals the current partition
            bool did_fracture = false;
            p.CheckDisruptorInvariants();
            p.OrderByPartitionThenDisruptor();

            size_t it = 0;
            while (it < p.elements.size()) {
                size_t pspan        = p.SpanPartition(it);
                size_t gspan        = p.SpanDisruptorAndPartition(it);
                size_t p_end = it + pspan;

                if (pspan == gspan) {
                    // Check if value changes for first time
                    while (it < p_end) {
                        BrittlePrismEle& e = p.elements[it];
                        if (e.original_partition == e.partition && e.disruptor != e.original_partition) {
                            reassign(e);
                            did_fracture = true;
                        }
                        it++;
                    }
                }else {
                    // Partition was given multiple different disruptors
                    did_fracture = true;
                    while (it < p_end) {
                        BrittlePrismEle& e = p.elements[it];
                        if (e.partition != e.disruptor) {
                            reassign(e);
                        }
                        it++;
                    }
                }

            }
            if (did_fracture) {
                p.fracture_count++;
                if (p.fracture_count > p.elements.size() * 2) {
                    throw std::runtime_error("Brittle prism invariant broken. Possible missing updates to by reassign callback.");
                }
            }
            return did_fracture;
        }
        static void Test0();
        static void Test1();
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    class GVN_Manager{

        struct GVN_Deps {
            GVN_Val op;
            std::vector<std::pair< GVN_Val, size_t> > args;
            void push(GVN_Val v, size_t count){args.emplace_back(v, count);}
            void push(GVN_Val v)              {args.emplace_back(v, 1);}

            void dump() {
                std::cout << "op: " << op << std::endl;
                for (size_t i = 0; i < args.size(); ++i) {
                    std::cout << "    [" <<i<< "]" << args[i].first << " : " << args[i].second << std::endl;
                }
            }
        };

        std::vector<GVN_Val>   builder_args_;
        std::optional<GVN_Val> builder_op_;
        std::optional<GVN_Val> builder_flags_;

        std::unordered_map< std::string,  GVN_Val > str_to_gvn_;     // Convenience map
        std::unordered_map< const void*, GVN_Val > ptr_to_gvn_;      // Convenience map
        std::unordered_map< size_t, GVN_Val > word_to_gvn_;          // Convenience map
        std::unordered_map< GVN_Val, std::optional<GVN_Deps> > gvn_;

    public:
        size_t stat_collisions;
        size_t stat_leaf_collisions;
        size_t stat_ca_too_big;

        size_t max_ca_size;

        GVN_Manager() : stat_collisions(0), stat_leaf_collisions(0), stat_ca_too_big(0), max_ca_size(32) {
            gvn_.insert({0, std::nullopt});
        }

        GVN_Val Leaf() {
            return Leaf(0);
        }
        GVN_Val Leaf(GVN_Val flags) {
            auto g = FindUnique(flags);
            gvn_.insert({g,std::nullopt});
            return g;
        }
        GVN_Manager& Op(GVN_Val op) {
            if (builder_op_){throw std::runtime_error("Multiple calls to Op(...) or missing End()");}
            builder_op_ = op;
            builder_flags_ = 0;
            return *this;
        }
        GVN_Manager& Arg(GVN_Val arg) {
            builder_args_.emplace_back(arg);
            return *this;
        }

        GVN_Val Args(const std::vector<GVN_Val>& args) {
            if (args.empty()){throw std::runtime_error("Args cannot be empty. Make empty argument lists explicit by passing a dummy leaf.");}
            builder_args_ = args;
            return End();
        }

        GVN_Val End() {
            if (!builder_op_) {throw std::runtime_error("Operator not specified");}
            if (builder_args_.empty()){throw std::runtime_error("Args not specified");}
            GVN_Val g = Create(*builder_op_, builder_args_);
            builder_flags_ = std::nullopt;
            builder_args_.clear();
            builder_op_ = std::nullopt;
            return g;
        }
        GVN_Val FromStr(const std::string& str) {return FromStr(str, 0);}
        GVN_Val FromStr(const std::string s, uint64_t flags)
        {
            auto q = str_to_gvn_.find(s);
            if (q == str_to_gvn_.end()) {
                str_to_gvn_.insert({s,Leaf(flags)});
                q = str_to_gvn_.find(s);
            }
            if (q == str_to_gvn_.end()) {throw std::runtime_error("Expected some value");}
            if ((q->second & GVN_MASK) != flags) {
                throw std::runtime_error("Inconsistent flags for literal: " + s);
            }
            return str_to_gvn_[s];
        }
        GVN_Val FromWord(size_t w)  {return FromWord(w, 0);}
        GVN_Val FromWord(size_t w, uint64_t flags)
        {
          auto q = word_to_gvn_.find(w);
          if (q == word_to_gvn_.end()) {
            word_to_gvn_.insert({w,Leaf(flags)});
            q = word_to_gvn_.find(w);
          }
          if (q == word_to_gvn_.end()) {throw std::runtime_error("Expected some value");}
          if ((q->second & GVN_MASK) != flags) {
            throw std::runtime_error("Inconsistent flags for gvn generated from word: " + std::to_string(w));
          }
          return word_to_gvn_[w];
        }
        GVN_Val FromPtr(const void* p) {return FromPtr(p, 0);}
        GVN_Val FromPtr(const void* p, uint64_t flags)
        {
            auto q = ptr_to_gvn_.find(p);
            if (q == ptr_to_gvn_.end()) {
                ptr_to_gvn_.insert({p,Leaf(flags)});
                q = ptr_to_gvn_.find(p);
            }
            if (q == ptr_to_gvn_.end()) {throw std::runtime_error("Expected some value");}
            if ((q->second & GVN_MASK) != flags) {
                throw std::runtime_error("Inconsistent flags for gvn generated from pointer: " + std::to_string(reinterpret_cast<size_t>(p)));
            }
            return ptr_to_gvn_[p];
        }

        GVN_Val Prism(GVN_Val op, BrittlePrism& brittle) {
            // It is sufficient to create a prism from the post array after taking the union
            //      with the pre array
            Op(op);
            brittle.OrderByPartition();
            size_t i = 0;
            while (i < brittle.elements.size()) {
                Arg(brittle.elements[i].partition);
                i += brittle.SpanPartition(i);
            }
            return End();
        }

    private:

        GVN_Val Create(GVN_Val op, const std::vector<GVN_Val>& args) {
            if (args.empty()){throw std::runtime_error("Logic error: GVN operator applied to zero args.");}
            std::optional<GVN_Deps> new_gvn = GVN_Deps();
            new_gvn->op = op;
            // Initialize new_gvn.args
            //      Either a copy of args or count of operator cluster leaves
            if (GVN_ValueIsCA_Op(new_gvn->op)) {
                std::vector<std::pair< GVN_Val, size_t> > acc;
                for (auto arg : args) {
                    if (GVN_ValueHasDeps(arg)) {
                        auto ad = *(gvn_[arg]);
                        if (ad.op == new_gvn->op) {
                            for (auto leaf : ad.args) {acc.emplace_back(leaf);}
                        }else {
                            acc.emplace_back(arg, 1);
                        }
                    }else {
                        acc.emplace_back(arg, 1);
                    }
                }
                std::sort(acc.begin(), acc.end());
                if (!acc.empty()) {
                    GVN_Val last  = GVN_NULL;
                    size_t acc_at = -1;
                    for (auto ele : acc) {
                        if (ele.first != last) {
                            acc_at = new_gvn->args.size();
                            new_gvn->args.emplace_back(ele.first, 0);
                        }
                        new_gvn->args[acc_at].second += ele.second;
                        last = ele.first;
                    }
                }
            }else {
                for (auto a : args) {new_gvn->push(a);}
            }
            // Handle the case where all args are the same
            // such as in      cond ? foo : foo
            //                 foo
            if (op & GVN_OP_IS_SWITCH) {
                bool all_same = true;
                if (args.size()) {
                    GVN_Val ele = args[0];
                    for (auto a : args) {
                        if (a != ele) {
                            all_same=false;
                            break;
                        }
                    }
                    if (all_same) {
                        // Do not create a new gvn, use the same as each 'case'
                        return ele;
                    }
                }
            }

            GVN_Val v = CalculateHash(*new_gvn);

            // Check if gvn is already in use, compare with existing gvn if so
            if (gvn_.find(v) != gvn_.end()) {
                if ( gvn_[v] ) {
                    auto prev = *(gvn_[v]);
                    if (!DepsEqual(prev, *new_gvn)) {
                        // Collision with internal node
                        v = FindUnique((v & GVN_MASK) | GVN_FROM_COLLISION );
                        stat_collisions++;
                    }
                }else {
                    // Collision between internal and leaf node
                    v = FindUnique((v & GVN_MASK) | GVN_FROM_COLLISION );
                    stat_leaf_collisions++;
                }
            }

            if (new_gvn->args.size() > max_ca_size) {
                // Avoid excess memory usage by returning a unique gvn
                //  and discarding all args
                v &= ~GVN_HAS_DEPS;
                v = FindUnique(v);
                new_gvn = std::nullopt;
                stat_ca_too_big++;
            }

            // ----------------- commit ---------------------
            if (gvn_.find(v) == gvn_.end()) {
                gvn_.insert({v,new_gvn});
            }
            return v;
        }

    private:
        GVN_Val FindUnique() {
            return FindUnique(0);
        }
        GVN_Val FindUnique(GVN_Val flags) {
            GVN_Val g;
            do{
                g = random() & ~GVN_MASK;
                g |= flags;
            }while(gvn_.find(g) != gvn_.end());

            return g;
        }

        GVN_Val CalculateHash(const GVN_Deps& deps) {
            // The lower bits are used to store properties for operations and
            //    keep track of context dependence of values
            GVN_Val flags = GVN_HAS_DEPS;
            for (auto arg : deps.args) {
                flags |= arg.first & GVN_MASK_INHERIT;
            }
            // Override GVN_IS_LOCAL_VALUE for operators introducing placeholders
            if (GVN_ValueParamDependent(deps.op))     {flags |= GVN_IS_LOCAL_VALUE;}
            if (GVN_ValueIsParamIndependent(deps.op)) {flags &= GVN_IS_LOCAL_VALUE;}

            GVN_Val v = 0;
            if (GVN_ValueIsCA_Op(deps.op)) {
                //std::cout << "--------------CA------------------"<<std::endl;
                /* Hash in the operator for edges entering a contiguous dag of the same operator
                 * Take the sum of hashes for internal nodes.
                 */

                for (size_t i = 0; i < deps.args.size(); ++i) {
                    if (GVN_ValueHasDeps(deps.args[i].first)) {
                        auto ad = *(gvn_[deps.args[i].first]);
                        if (ad.op == deps.op) {
                            v += deps.args[i].first;
                        }else {
                            v += deps.args[i].first * deps.op;
                        }
                    }else {
                        v += deps.args[i].first * deps.op;
                    }
                }
            }else {
                //std::cout << "+++++++++++++++STANDARD HASHER+++++++++++++++++"<<std::endl;
                for (size_t i = 0; i < deps.args.size(); i++) {
                    v ^= deps.args[i].first * (i+1);
                }
            }
            v = (v & ~GVN_MASK) | flags;
            return v;
        }

        static bool DepsEqual(GVN_Deps& a, GVN_Deps& b) {
            if (a.op != b.op) {return false;}
            if (a.args.size() != b.args.size()) {return false;}
            for (size_t i = 0; i < a.args.size(); i++) {
                if (a.args[i] != b.args[i]) {return false;}
            }
            return true;
        }

    public:
        static void Test0();
        static void Test1();
        static void Test2();
        static void Test3();
    };

    inline void RunAllTests()
    {
        BrittlePrism::Test0();
        BrittlePrism::Test1();
        GVN_Manager::Test0();
        GVN_Manager::Test1();
        GVN_Manager::Test2();
        GVN_Manager::Test3();
    }
};

#endif

#endif // JLM_GVN_H
