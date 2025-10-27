//
// Created by lars-astrup-sundt on 10/27/25.
//

#include "gvn.hpp"

//
// Created by lars-astrup-sundt on 10/26/25.
//

bool jlm::rvsdg::gvn::gvn_verbose = false;
namespace jlm::rvsdg::gvn {


    void GVN_Manager::Test0() {
        GVN_Manager gm;
        auto my_op = gm.Leaf();
        auto my_ca = gm.Leaf(GVN_OP_IS_CA);

        auto x = gm.Leaf();
        auto y = gm.Leaf();

        if (x == y) {
            throw std::runtime_error("Expected unique hashes");
        }
        if (gvn_verbose) {
            std::cout << "X: " << to_string(x) << std::endl;
            std::cout << "Y: " << to_string(y) << std::endl;

            std::cout << "MyOp: " << to_string(my_op) << std::endl;
            std::cout << "MyCa: " << to_string(my_ca) << std::endl;
        }
        {
            GVN_Deps d;
            d.op = my_op;
            d.push(x);
            d.push(y);
            if (gvn_verbose){std::cout << "Hash of x - y is : " << to_string(gm.CalculateHash(d)) << std::endl;}
        }
        {
            GVN_Deps d2;
            d2.op = my_op;
            d2.push(y);
            d2.push(x);
            if (gvn_verbose){std::cout << "Hash of y - x is : " << to_string(gm.CalculateHash(d2)) << std::endl;}
        }
        {
            GVN_Deps d3;
            d3.op = my_ca;
            d3.push(y);
            d3.push(x);
            gm.gvn_.insert({gm.CalculateHash(d3), d3});

            GVN_Deps d4;
            d4.op = my_ca;
            d4.push(y);
            d4.push(gm.CalculateHash(d3));

            GVN_Deps d5;
            d5.op = my_ca;
            d5.push(gm.CalculateHash(d3));
            d5.push(y);

            if (gm.CalculateHash(d4) != gm.CalculateHash(d5)) {
                throw std::runtime_error("Test failed: expected equal hashes from associative and commutative op");
            }
            if (gvn_verbose) {
                std::cout << "y+x " << to_string(gm.CalculateHash(d3)) << std::endl;
                std::cout << "y + (y+x): " << to_string(gm.CalculateHash(d4)) << std::endl;
                std::cout << "(y+x) + y: " << to_string(gm.CalculateHash(d5)) << std::endl;
            }
        }
    }
    void GVN_Manager::Test1()
    {
        GVN_Manager gm;
        const char* str = "foo";
        GVN_Val a = gm.FromPtr(str);
        GVN_Val b = gm.FromPtr(str);
        if (a != b) {
            throw std::runtime_error("FromPtr failed");
        }
    }
    void GVN_Manager::Test2()
    {
        GVN_Manager gm;
        GVN_Val x = gm.Leaf();
        GVN_Val y = gm.Leaf();

        GVN_Val switchy_op = gm.FromPtr("switch", GVN_OP_IS_SWITCH);

        GVN_Val xx   = gm.Op(switchy_op).Args({x,x});
        std::cout << "*************************"<< std::endl << std::endl;
        GVN_Val xy   = gm.Op(switchy_op).Args({x,y});

        std::cout << "*************************"<< std::endl << std::endl;
        GVN_Val xy_2 = gm.Op(switchy_op).Args({x,y});

        if (xy != xy_2) {
            std::cout << to_string(xy) << to_string(xy_2) << std::endl;
            throw std::runtime_error("Values should be stable");
        }

        if (xy == xx) {throw std::runtime_error("Values should be different");}
        if (xx != x)  {throw std::runtime_error("Values should be same");}
    }

    void GVN_Manager::Test3()
    {
        GVN_Manager gm;
        auto a   = gm.Leaf();
        auto b   = gm.Leaf();
        auto c   = gm.Leaf();
        auto d   = c;

        auto op_add             = gm.FromStr("+", GVN_OP_IS_CA);
        auto op_alternatives    = gm.FromStr("merge", GVN_OP_IS_SWITCH);
        auto one                = gm.FromStr("1");

        auto reassign = [&gm, op_alternatives](BrittlePrismEle& ele) {
            ele.partition = gm.Op(op_alternatives).Args({ele.disruptor, ele.partition});
        };

        // loop  (
        //          a + 1      ->   a
        //          b + a      ->   b
        //              c      ->   d
        //              d      ->   c
        // )

        GVN_Val last = 0;
        for (size_t k = 0; k < 4; k++) {
            std::optional< BrittlePrism > prevars  = std::nullopt;
            std::optional< BrittlePrism > postvars = std::nullopt;

            GVN_Val a_out = 0;
            GVN_Val b_out = 0;
            GVN_Val c_out = 0;
            GVN_Val d_out = 0;
            size_t max_iter = 100;

            std::cout << "INIT: " << a_out << " - " << b_out << " - " << c_out << " - " << d_out << std::endl;

            while (max_iter) {
                max_iter--;
                if (!prevars) {
                    prevars = BrittlePrism({a,b,c,d});
                }else {
                    prevars->elements[0].disruptor = a_out;
                    prevars->elements[1].disruptor = b_out;
                    prevars->elements[2].disruptor = c_out;
                    prevars->elements[3].disruptor = d_out;

                    if (! BrittlePrism::Fracture(*prevars,reassign) ){break;}
                }
                a_out = gm.Op(op_add).Args({a, one});
                b_out = gm.Op(op_add).Args({b,a});
                c_out = d;
                d_out = c;
                if (!postvars) {
                    postvars = BrittlePrism( {a_out, b_out, c_out, d_out});
                }else {
                    postvars->elements[0].disruptor = a_out;
                    postvars->elements[1].disruptor = b_out;
                    postvars->elements[2].disruptor = c_out;
                    postvars->elements[3].disruptor = d_out;
                    if (! BrittlePrism::Fracture(*postvars,reassign) ){break;}
                }
            }

            std::cout << "after:    " << a_out << " - " << b_out << " - " << c_out << " - " << d_out << std::endl;

            GVN_Val h = gm.Op(op_alternatives).Args({a_out,b_out,c_out,d_out});
            if (last && h != last && !(h & GVN_FROM_COLLISION)) {
                std::cout << to_string(last) << " " << to_string(h) << std::endl;
                throw std::runtime_error("Hashes where different across iterations");
            }
            last = h;
        }
    }

    void BrittlePrism::Test0()
    {
        std::vector<GVN_Val> v = {77,128,128,77,77};
        BrittlePrism p0(v);
        p0.dump();
        p0.OrderByPartition();
        if (gvn_verbose){std::cout << "----------partitions-------------"<<std::endl;}
        p0.dump();
        p0.OrderByOriginal();
        if (gvn_verbose){std::cout << "----------original---------------"<<std::endl;}
        p0.dump();

        p0.elements[2].disruptor = 198;
        p0.elements[0].disruptor = 199;
        p0.elements[3].disruptor = 199;
        p0.elements[4].disruptor = 199;

        p0.OrderByPartitionThenDisruptor();
        if (gvn_verbose){std::cout << "----------before fracture---------------"<<std::endl;}
        p0.dump();
        if (gvn_verbose){std::cout << "-----------------------------------"<<std::endl;}
        BrittlePrism::Fracture(p0, [](BrittlePrismEle& ele) {
            if (gvn_verbose){std::cout << "Reassign " << ele.partition << "->" << ele.disruptor << std::endl;}
            ele.partition = ele.disruptor;
        });
        p0.OrderByPartition();
        if (gvn_verbose){std::cout << "----------after frac---------------"<<std::endl;}
        p0.dump();

        p0.dump_partitions();

        p0.OrderByPartition();
        p0.elements[0].disruptor = 1001;
        p0.elements[p0.elements.size()-1].disruptor = 1001;

        try {
            BrittlePrism::Fracture(p0, [](BrittlePrismEle& ele) {
                if (gvn_verbose){std::cout << "Reassign " << ele.partition << "->" << ele.disruptor << std::endl;}
                ele.partition = ele.disruptor;
            });
        }catch (std::runtime_error& e) {
            if (gvn_verbose){std::cout << "success : should throw : " << e.what() << std::endl;}
        }

        BrittlePrism::EachPartition(p0, [](BrittlePrismEle& ele, size_t count) {
            std::cout << ele.partition << " : " << count << std::endl;
        });


        BrittlePrism p2 = p0;
        p2 = p0;
        p0.elements[0].disruptor = 10;
        if (gvn_verbose){std::cout << "=======================================" << std::endl;}
        p0.dump();
        if (gvn_verbose){std::cout << "=======================================" << std::endl;}
        p2.dump();
        if (gvn_verbose){std::cout << "=======================================" << std::endl;}

        auto shatter_good = [](BrittlePrismEle& ele, size_t index) {
            ele.partition = index;
        };
        auto shatter_bad = [](BrittlePrismEle& ele, size_t index) {
            ele.partition = 88;
        };
        Shatter(p0, shatter_good);
        try {
            Shatter(p2, shatter_bad);
        }catch (std::runtime_error& e) {
            if (gvn_verbose){std::cout << "success : should throw : detected bad shatter : " << e.what() << std::endl;}
        }
    }

    void BrittlePrism::Test1()
    {
        auto br = BrittlePrism({1,1,1,4});
        br.elements[0].disruptor = 10;
        br.elements[1].disruptor = 10;
        br.elements[2].disruptor = 88;
        br.elements[3].disruptor = 77;
        BrittlePrism::Fracture(br, [](BrittlePrismEle& ele) {
            ele.partition = ele.disruptor;
        });
        br.OrderByOriginal();
        if (gvn_verbose){br.dump();}
        if (br.elements[0].partition != 10){throw std::runtime_error("should be 10");}
        if (br.elements[1].partition != 10){throw std::runtime_error("should be 10");}
        if (br.elements[2].partition != 88){throw std::runtime_error("should be 88");}
        if (br.elements[3].partition != 77){throw std::runtime_error("should be 77");}
        ////////////////////////////////////////////////////////////
        br.elements[0].disruptor = 100;
        br.elements[1].disruptor = 100;
        br.elements[2].disruptor = 888;
        br.elements[3].disruptor = 123;
        BrittlePrism::Fracture(br, [](BrittlePrismEle& ele) {
            ele.partition = ele.disruptor;
        });
        br.OrderByOriginal();
        if (gvn_verbose){br.dump();}
        if (br.elements[0].partition != 10){throw std::runtime_error("should still be 10");}
        if (br.elements[1].partition != 10){throw std::runtime_error("should still be 10");}
        if (br.elements[2].partition != 88){throw std::runtime_error("should be 88");}
        if (br.elements[3].partition != 77){throw std::runtime_error("should be 77");}

        ////////////////////////////////////////////////////////////
        br.elements[0].disruptor = 1780;
        br.elements[1].disruptor = 10;
        br.elements[2].disruptor = 81488;
        br.elements[3].disruptor = 12153;
        BrittlePrism::Fracture(br, [](BrittlePrismEle& ele) {
            ele.partition = ele.disruptor;
        });
        br.OrderByOriginal();
        if (gvn_verbose){br.dump();}
        if (br.elements[0].partition != 1780){throw std::runtime_error("should be 1780");}
        if (br.elements[1].partition != 10){throw std::runtime_error("should still be 10");}
        if (br.elements[2].partition != 88){throw std::runtime_error("should still be 88");}
        if (br.elements[3].partition != 77){throw std::runtime_error("should still be 77");}
    }
}

