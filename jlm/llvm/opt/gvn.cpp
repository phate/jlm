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
        auto my_ca = gm.Leaf(GVN_OPERATOR_IS_CA);

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
            if (gvn_verbose){std::cout << "Hash of x - y is : " << to_string(gm.CalculateHash(d).first) << std::endl;}
        }
        {
            GVN_Deps d2;
            d2.op = my_op;
            d2.push(y);
            d2.push(x);
            if (gvn_verbose){std::cout << "Hash of y - x is : " << to_string(gm.CalculateHash(d2).first) << std::endl;}
        }
        {
            GVN_Deps d3;
            d3.op = my_ca;
            d3.push(y);
            d3.push(x);
            gm.gvn_.insert({gm.CalculateHash(d3).first, d3});

            GVN_Deps d4;
            d4.op = my_ca;
            d4.push(y);
            d4.push(gm.CalculateHash(d3).first);

            GVN_Deps d5;
            d5.op = my_ca;
            d5.push(gm.CalculateHash(d3).first);
            d5.push(y);

            if (gm.CalculateHash(d4).first != gm.CalculateHash(d5).first) {
                throw std::runtime_error("Test failed: expected equal hashes from associative and commutative op");
            }
            if (gvn_verbose) {
                std::cout << "y+x " << to_string(gm.CalculateHash(d3).first) << std::endl;
                std::cout << "y + (y+x): " << to_string(gm.CalculateHash(d4).first) << std::endl;
                std::cout << "(y+x) + y: " << to_string(gm.CalculateHash(d5).first) << std::endl;
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

        GVN_Val xx   = gm.Op(GVN_OP_ANY_ORDERED).Args({x,x});
        std::cout << "*************************"<< std::endl << std::endl;
        GVN_Val xy   = gm.Op(GVN_OP_ANY_ORDERED).Args({x,y});

        std::cout << "*************************"<< std::endl << std::endl;
        GVN_Val xy_2 = gm.Op(GVN_OP_ANY_ORDERED).Args({x,y});

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

        auto op_add             = gm.FromStr("+", GVN_OPERATOR_IS_CA);
        auto one                = gm.FromStr("1");

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

                    if (! prevars->Fracture() ){break;}
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
                    if (! postvars->Fracture() ){break;}
                }
            }

            std::cout << "after:    " << a_out << " - " << b_out << " - " << c_out << " - " << d_out << std::endl;

            GVN_Val h = gm.Op(GVN_OP_ANY_ORDERED).Args({a_out,b_out,c_out,d_out});
            if (last && h != last && !(h & GVN_FROM_COLLISION)) {
                std::cout << to_string(last) << " " << to_string(h) << std::endl;
                throw std::runtime_error("Hashes where different across iterations");
            }
            last = h;
        }
    }

    void GVN_Manager::Test4()
    {
        GVN_Manager gm;
        GVN_Val four = gm.FromWord(4);
        GVN_Val zero = gm.FromWord(0);
        GVN_Val too_big = gm.FromWord(0xffffffffffffffull);
        GVN_Val same_too_big = gm.FromWord(0xffffffffffffffull);
        if (four != 4){throw std::runtime_error("Small value not non-symbolic: " + to_string(four));}
        if (zero != 0){throw std::runtime_error("Small value not non-symbolic: " + to_string(zero));}
        if (!(too_big & GVN_IS_SYMBOLIC)){throw std::runtime_error("Big constants should be symbols");}
        if (too_big != same_too_big){throw std::runtime_error("FromWord should be referentially transparent.");}
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
        p0.Fracture();
        p0.OrderByPartition();
        if (gvn_verbose){std::cout << "----------after frac---------------"<<std::endl;}
        p0.dump();

        p0.dump_partitions();

        p0.OrderByPartition();
        p0.elements[0].disruptor = 1001;
        p0.elements[p0.elements.size()-1].disruptor = 1001;

        try {
            p0.Fracture();
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
        br.dump();
        std::cout << "*****************************************" << std::endl;
        br.elements[0].disruptor = 10;
        br.elements[1].disruptor = 10;
        br.elements[2].disruptor = 88;
        br.elements[3].disruptor = 77;
        br.Fracture();
        br.dump();
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
        br.Fracture();
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
        br.Fracture();
        br.OrderByOriginal();
        if (gvn_verbose){br.dump();}
        if (br.elements[0].partition != 1780){throw std::runtime_error("should be 1780");}
        if (br.elements[1].partition != 10){throw std::runtime_error("should still be 10");}
        if (br.elements[2].partition != 88){throw std::runtime_error("should still be 88");}
        if (br.elements[3].partition != 77){throw std::runtime_error("should still be 77");}
    }

    void GVN_Manager::Test5()
    {
        if (GVN_MASK & GVN_SMALL_VALUE) {
            throw std::runtime_error("Flags of gvn value cannot overlap with field for integer constants.");
        }

        GVN_Manager gm;
        auto forty = gm.Op(GVN_OP_ADDITION).Arg(10).Arg(30).End();
        if (forty != 40){throw std::runtime_error("Bad addition" + to_string(forty));}
        auto twelve = gm.Op(GVN_OP_MULTIPLY).Arg(4).Arg(3).End();
        if (twelve != 12){throw std::runtime_error("Bad multiplication");}
    }

    void GVN_Manager::Test6() {
        GVN_Manager gm;
        if (gm.Op(GVN_OP_EQ).Arg(10).Arg(242).End() != GVN_FALSE) {
            throw std::runtime_error("Should have been false");
        }
        if (gm.Op(GVN_OP_EQ).Arg(20).Arg(20).End() != GVN_TRUE) {
            throw std::runtime_error("Should have been true");
        }
        auto l = gm.Leaf();
        if (gm.Op(GVN_OP_EQ).Arg(l).Arg(l).End() != GVN_TRUE) {
            throw std::runtime_error("Equal symbols must be true");
        }
        if (gm.Op(GVN_OP_NEQ).Arg(24).Arg(l).End() & GVN_CONST_SYMBOL) {
            throw std::runtime_error("NEQ bad inference.");
        }
        if (gm.Op(GVN_OP_NEQ).Arg(24).Arg(24).End() != GVN_FALSE) {
            throw std::runtime_error("NEQ bad inference.");
        }
        if (gm.Op(GVN_OP_NEQ).Arg(214).Arg(24).End() != GVN_TRUE) {
            throw std::runtime_error("NEQ bad inference.");
        }
    }
}

