"""Tests for Hyperparameter descriptor class."""

import pytest

from climatrix.utils.hyperparameter import Hyperparameter


class TestHyperparameterDescriptor:
    """Test the Hyperparameter descriptor class."""

    def test_basic_initialization(self):
        """Test basic hyperparameter initialization."""
        hp = Hyperparameter(int)
        assert hp.param_type == int
        assert hp.bounds is None
        assert hp.values is None
        assert hp.default is None

    def test_initialization_with_bounds(self):
        """Test hyperparameter initialization with bounds."""
        hp = Hyperparameter(float, bounds=(0.0, 10.0), default=5.0)
        assert hp.param_type == float
        assert hp.bounds == (0.0, 10.0)
        assert hp.default == 5.0

    def test_initialization_with_values(self):
        """Test hyperparameter initialization with categorical values."""
        hp = Hyperparameter(str, values=["fast", "slow"], default="fast")
        assert hp.param_type == str
        assert hp.values == ["fast", "slow"]
        assert hp.default == "fast"

    def test_initialization_bounds_and_values_error(self):
        """Test that specifying both bounds and values raises error."""
        with pytest.raises(ValueError, match="Cannot specify both bounds and values"):
            Hyperparameter(int, bounds=(1, 10), values=[1, 2, 3])

    def test_bounds_non_numeric_type_error(self):
        """Test that bounds with non-numeric type raises error."""
        with pytest.raises(ValueError, match="Bounds can only be specified for numeric types"):
            Hyperparameter(str, bounds=(1, 10))

    def test_invalid_bounds_length_error(self):
        """Test that bounds with wrong length raises error."""
        with pytest.raises(ValueError, match="Bounds must be a tuple of \\(min_value, max_value\\)"):
            Hyperparameter(int, bounds=(1, 2, 3))

    def test_invalid_bounds_order_error(self):
        """Test that bounds with min >= max raises error."""
        with pytest.raises(ValueError, match="Lower bound must be less than upper bound"):
            Hyperparameter(int, bounds=(10, 5))

        with pytest.raises(ValueError, match="Lower bound must be less than upper bound"):
            Hyperparameter(float, bounds=(5.0, 5.0))

    def test_set_name_protocol(self):
        """Test the __set_name__ descriptor protocol."""
        hp = Hyperparameter(int)
        hp.__set_name__(None, "test_param")
        assert hp.name == "test_param"
        assert hp.private_name == "_test_param"


class TestHyperparameterInClass:
    """Test hyperparameter descriptor when used in a class."""

    def setup_method(self):
        """Set up test class with hyperparameters."""
        class TestClass:
            power = Hyperparameter(float, bounds=(0.5, 5.0), default=2.0)
            k = Hyperparameter(int, bounds=(1, 20), default=5)
            mode = Hyperparameter(str, values=["fast", "slow"], default="fast")
            optional = Hyperparameter(int)

        self.TestClass = TestClass

    def test_default_values(self):
        """Test that default values are returned correctly."""
        obj = self.TestClass()
        assert obj.power == 2.0
        assert obj.k == 5
        assert obj.mode == "fast"
        assert obj.optional is None

    def test_setting_valid_values(self):
        """Test setting valid values."""
        obj = self.TestClass()
        
        obj.power = 3.5
        assert obj.power == 3.5
        
        obj.k = 10
        assert obj.k == 10
        
        obj.mode = "slow"
        assert obj.mode == "slow"

    def test_type_casting(self):
        """Test automatic type casting."""
        obj = self.TestClass()
        
        # String to float
        obj.power = "3.5"
        assert obj.power == 3.5
        assert isinstance(obj.power, float)
        
        # String to int
        obj.k = "15"
        assert obj.k == 15
        assert isinstance(obj.k, int)
        
        # Float to int (truncation)
        obj.k = 12.7
        assert obj.k == 12
        assert isinstance(obj.k, int)

    def test_bounds_validation(self):
        """Test bounds validation."""
        obj = self.TestClass()
        
        # Valid bounds
        obj.power = 1.0
        assert obj.power == 1.0
        
        obj.power = 5.0  # Upper bound
        assert obj.power == 5.0
        
        obj.power = 0.5  # Lower bound
        assert obj.power == 0.5
        
        # Invalid bounds
        with pytest.raises(ValueError, match="outside bounds \\[0.5, 5.0\\]"):
            obj.power = 0.4
        
        with pytest.raises(ValueError, match="outside bounds \\[0.5, 5.0\\]"):
            obj.power = 5.1
        
        # Integer bounds
        obj.k = 1  # Lower bound
        assert obj.k == 1
        
        obj.k = 20  # Upper bound
        assert obj.k == 20
        
        with pytest.raises(ValueError, match="outside bounds \\[1, 20\\]"):
            obj.k = 0
        
        with pytest.raises(ValueError, match="outside bounds \\[1, 20\\]"):
            obj.k = 21

    def test_categorical_validation(self):
        """Test categorical values validation."""
        obj = self.TestClass()
        
        # Valid values
        obj.mode = "fast"
        assert obj.mode == "fast"
        
        obj.mode = "slow"
        assert obj.mode == "slow"
        
        # Invalid value
        with pytest.raises(ValueError, match="not in valid values \\['fast', 'slow'\\]"):
            obj.mode = "medium"

    def test_type_conversion_errors(self):
        """Test type conversion errors."""
        obj = self.TestClass()
        
        # Cannot convert to float
        with pytest.raises(TypeError, match="Cannot convert .* to float"):
            obj.power = "not_a_number"
        
        # Cannot convert to int
        with pytest.raises(TypeError, match="Cannot convert .* to int"):
            obj.k = "not_a_number"

    def test_none_values(self):
        """Test handling of None values."""
        obj = self.TestClass()
        
        # Setting None should use default
        obj.power = None
        assert obj.power == 2.0
        
        obj.k = None
        assert obj.k == 5
        
        # For parameter without default, None should remain None
        obj.optional = None
        assert obj.optional is None

    def test_descriptor_on_class(self):
        """Test accessing descriptor on class returns the descriptor."""
        assert isinstance(self.TestClass.power, Hyperparameter)
        assert isinstance(self.TestClass.k, Hyperparameter)
        assert isinstance(self.TestClass.mode, Hyperparameter)

    def test_get_spec_method(self):
        """Test the get_spec method."""
        power_spec = self.TestClass.power.get_spec()
        expected_power = {
            "type": float,
            "bounds": (0.5, 5.0),
            "default": 2.0
        }
        assert power_spec == expected_power
        
        k_spec = self.TestClass.k.get_spec()
        expected_k = {
            "type": int,
            "bounds": (1, 20),
            "default": 5
        }
        assert k_spec == expected_k
        
        mode_spec = self.TestClass.mode.get_spec()
        expected_mode = {
            "type": str,
            "values": ["fast", "slow"],
            "default": "fast"
        }
        assert mode_spec == expected_mode
        
        optional_spec = self.TestClass.optional.get_spec()
        expected_optional = {"type": int}
        assert optional_spec == expected_optional


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_boolean_type(self):
        """Test boolean hyperparameters."""
        class TestClass:
            flag = Hyperparameter(bool, default=True)
        
        obj = TestClass()
        assert obj.flag is True
        
        obj.flag = False
        assert obj.flag is False
        
        # Test type casting
        obj.flag = 0
        assert obj.flag is False
        
        obj.flag = 1
        assert obj.flag is True
        
        obj.flag = "True"
        assert obj.flag is True

    def test_complex_numeric_types(self):
        """Test with different numeric types."""
        class TestClass:
            small_int = Hyperparameter(int, bounds=(-10, 10), default=0)
            big_float = Hyperparameter(float, bounds=(1e-6, 1e6), default=1.0)
        
        obj = TestClass()
        
        # Test negative bounds
        obj.small_int = -5
        assert obj.small_int == -5
        
        # Test scientific notation
        obj.big_float = 1e5
        assert obj.big_float == 1e5
        
        with pytest.raises(ValueError, match="outside bounds"):
            obj.small_int = -11
        
        with pytest.raises(ValueError, match="outside bounds"):
            obj.big_float = 1e7

    def test_list_and_tuple_categorical(self):
        """Test categorical values with different collection types."""
        class TestClass:
            list_param = Hyperparameter(str, values=["a", "b", "c"])
            mixed_param = Hyperparameter(int, values=[1, 2, 3])
        
        obj = TestClass()
        
        obj.list_param = "a"
        assert obj.list_param == "a"
        
        obj.mixed_param = 2
        assert obj.mixed_param == 2
        
        # Type casting should work with categorical too
        obj.mixed_param = "3"
        assert obj.mixed_param == 3

    def test_multiple_instances(self):
        """Test that different instances maintain separate values."""
        class TestClass:
            value = Hyperparameter(int, default=5)
        
        obj1 = TestClass()
        obj2 = TestClass()
        
        obj1.value = 10
        obj2.value = 20
        
        assert obj1.value == 10
        assert obj2.value == 20

    def test_inheritance(self):
        """Test hyperparameters with class inheritance."""
        class BaseClass:
            base_param = Hyperparameter(int, default=1)
        
        class DerivedClass(BaseClass):
            derived_param = Hyperparameter(float, default=2.0)
        
        obj = DerivedClass()
        assert obj.base_param == 1
        assert obj.derived_param == 2.0
        
        obj.base_param = 5
        obj.derived_param = 3.5
        
        assert obj.base_param == 5
        assert obj.derived_param == 3.5