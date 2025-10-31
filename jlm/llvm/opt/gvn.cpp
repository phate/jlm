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
        // TODO: arithmentic tests
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
        auto a = gm.Leaf();
        auto b = gm.FromWord(88);
        if ( gm.Op(GVN_OP_ADDITION).Args({a,b}) != gm.Op(GVN_OP_ADDITION).Args({b,GVN_IGNORE,a,GVN_IGNORE}) ) {
            throw std::runtime_error("Should have ignored some args");
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
        GVN_Manager gm;
        auto forty = gm.Op(GVN_OP_ADDITION).Arg(10).Arg(30).End();
        if (forty != 40){throw std::runtime_error("Bad addition" + to_string(forty));}
        auto twelve = gm.Op(GVN_OP_MULTIPLY).Arg(4).Arg(3).End();
        if (twelve != 12){throw std::runtime_error("Bad multiplication");}

        auto x = gm.Leaf();
        auto x_plus_zero = gm.Op(GVN_OP_ADDITION).Arg(0).Arg(x).End();
        auto x_times_zero = gm.Op(GVN_OP_MULTIPLY).Arg(0).Arg(x).End();
        auto x_times_one = gm.Op(GVN_OP_MULTIPLY).Arg(1).Arg(x).End();

        auto x2 = gm.Op(GVN_OP_MULTIPLY).Arg(2).End();
        if (x2 == x){throw std::runtime_error("x == 2*x");}
        if (x_plus_zero != x){throw std::runtime_error("Bad addition x + 0  != x");}
        if (x_times_zero != 0){throw std::runtime_error("Bad multiplication x * 0 != 0");}
        if (x_times_one != x){throw std::runtime_error("Bad multiplication x * 1 != x");}
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

